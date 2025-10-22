import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import sys
import os
sys.path.append("../utility/")
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
from Utility import data_collecter
import time

# 定义ODE流形拟合网络
class ODEManifold(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ODEManifold, self).__init__()
        self.input_dim = input_dim
        # ODE向量场参数化网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        # 确保向量场在流形上
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        """计算向量场 F(x)"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.norm(x)  # 归一化确保平滑性
    
    def integrate(self, x0, t):
        """使用欧拉方法积分ODE"""
        x = x0.clone()
        dt = t[1] - t[0] if len(t) > 1 else t[0]
        
        for _ in range(len(t)-1):
            dx = self.forward(x)
            x = x + dx * dt
        return x
    
    def geodesic_distance(self, x, y, num_steps=100):
        """
        正确计算流形上两点间的测地距离（单步积分，维度严格对齐）
        Args:
            x: 输入点张量，形状 [batch_size, state_dim] → 例：(50, 2)
            y: 目标点张量，形状 [batch_size, state_dim] → 例：(50, 2)
            num_steps: 时间插值步数（控制积分精度）→ 例：100
        Returns:
            dist: 测地距离张量，形状 [batch_size] → 例：(50,)，每个元素对应x[i]到y[i]的距离
        """
        # 1. 维度一致性预检（提前规避不匹配问题）
        if x.shape != y.shape:
            raise ValueError(f"输入点维度不匹配！x.shape={x.shape}, y.shape={y.shape}")
        if x.dim() != 2:
            raise ValueError(f"输入点必须是2维张量（[批次数, 状态维]），当前x.dim()={x.dim()}")
        device = x.device
        batch_size, state_dim = x.shape  
        dt = 1.0 / (num_steps - 1)      
        t = torch.linspace(0, 1, num_steps, device=device)
        path = x.unsqueeze(0) + t.unsqueeze(1).unsqueeze(2) * (y - x).unsqueeze(0)
        velocities = self.forward(path.reshape(-1, state_dim)).reshape(num_steps, batch_size, state_dim)
        step_distances = torch.norm(velocities, dim=2)
        
        dist = torch.trapz(step_distances, t, dim=0)
        
        return dist  # 直接返回积分结果，无需二次处理

# 定义网络初始化函数
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self, encode_layers, belayers, bdlayers, Nkoopman, u_dim, control_encode_dim):
        super(Network, self).__init__()
        ELayers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            ELayers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                ELayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(ELayers)
        
        BELayers = OrderedDict()
        for layer_i in range(len(belayers)-1):
            BELayers["linear_{}".format(layer_i)] = nn.Linear(belayers[layer_i], belayers[layer_i+1])
            if layer_i != len(belayers)-2:
                BELayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.control_encoder = nn.Sequential(BELayers)  

        BDLayers = OrderedDict()
        for layer_i in range(len(bdlayers)-1):
            BDLayers["linear_{}".format(layer_i)] = nn.Linear(bdlayers[layer_i], bdlayers[layer_i+1])
            if layer_i != len(bdlayers)-2:
                BDLayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.control_decoder = nn.Sequential(BDLayers)  
    
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        # self.lB = nn.Linear(belayers[-1], Nkoopman, bias=False)
        self.lB = nn.Linear(control_encode_dim, Nkoopman, bias=False)
    # 状态提升
    def encode(self, x):
        return torch.cat([x, self.encode_net(x)], axis=-1)
    
    # 控制编码
    def control_encode(self, x, u):
        y = torch.cat([x, u], axis=-1) 
        gy = self.control_encoder(y)
        return torch.cat([y, gy], axis=-1)

    def control_decode(self, x_e):
        return self.control_decoder(x_e)

    def forward(self, z, hat_u):
        return self.lA(z) + self.lB(hat_u)

# 计算Kloss
def K_loss(data, net, u_dim=1, Nstate=4):
    steps, train_traj_num, Nstates = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    X_current = net.encode(data[0,:,u_dim:])
    max_loss_list = []
    mean_loss_list = []
    for i in range(steps-1):
        bilinear = net.bicode(X_current[:,:Nstate].detach(), data[i,:,:u_dim])
        X_current = net.forward(X_current, bilinear)
        Y = data[i+1,:,u_dim:]
        Err = X_current[:,:Nstate] - Y
        max_loss_list.append(torch.mean(torch.max(torch.abs(Err), axis=0).values).detach().cpu().numpy())
        mean_loss_list.append(torch.mean(torch.mean(torch.abs(Err), axis=0)).detach().cpu().numpy())
    return np.array(max_loss_list), np.array(mean_loss_list)

# 带流形约束的损失函数
def Klinear_loss_with_manifold(data, net, state_manifold, control_manifold, mse_loss, u_dim=1, gamma=0.99, 
                               Nstate=4, all_loss=0, detach=0, lambda_geom=0.1, lambda_lin=0.1, lambda_recon=0.1):
    steps, train_traj_num, NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    
    # 提取状态数据用于流形约束
    states = data[:, :, u_dim:]
    controls = data[:, :, :u_dim]
    batch_size = states.shape[1]
    
    # 计算流形几何约束损失
    geom_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    if lambda_geom > 0 and batch_size > 1:
        # 随机选择一些点对计算距离约束
        idx1 = torch.randint(0, batch_size, (min(100, batch_size//2),), device=device)
        idx2 = torch.randint(0, batch_size, (min(100, batch_size//2),), device=device)

        x_samples = states[0, idx1, :]
        y_samples = states[0, idx2, :]
        u1_samples = controls[0, idx1, :]
        u2_samples = controls[0, idx2, :]

        # 计算流形上的测地距离
        state_geo_dist = state_manifold.geodesic_distance(x_samples, y_samples)
        control_geo_dist = control_manifold.geodesic_distance(u1_samples, u2_samples)

        # 计算编码空间中的距离
        encoded_x = net.encode(x_samples)
        encoded_y = net.encode(y_samples)
        state_encoded_dist = torch.norm(encoded_x - encoded_y, dim=1)
        
        encoded_u1 = net.control_encode(x_samples, u1_samples)
        encoded_u2 = net.control_encode(y_samples, u2_samples)
        control_encoded_dist = torch.norm(encoded_u1 - encoded_u2, dim=1)

        # 计算比例因子c，使编码距离与测地距离成比例
        state_c = torch.sum(state_encoded_dist * state_geo_dist) / (torch.sum(state_geo_dist ** 2) + 1e-8) 
        control_c = torch.sum(control_encoded_dist * control_geo_dist) / (torch.sum(control_geo_dist ** 2) + 1e-8) 
        # 几何约束损失
        geom_loss = torch.mean(torch.abs(state_encoded_dist - state_c * state_geo_dist)) + torch.mean(torch.abs(control_encoded_dist - control_c * control_geo_dist))
    
    # 原始Koopman损失计算
    X_current = net.encode(data[0,:,u_dim:])
    beta = 1.0
    beta_sum = 0.0
    pred_loss = torch.zeros(1, dtype=torch.float64).to(device)
    lin_loss = torch.zeros(1, dtype=torch.float64).to(device)
    recon_loss = torch.zeros(1, dtype=torch.float64).to(device)
    
    for i in range(steps-1):
        hat_u = net.control_encode(X_current[:,:Nstate].detach() if detach else X_current[:,:Nstate], 
                             data[i,:,:u_dim])
        X_next = net.forward(X_current, hat_u)
        
        beta_sum += beta
        if not all_loss:
            pred_loss += beta * mse_loss(X_next[:,:Nstate], data[i+1,:,u_dim:])
        else:
            Y = net.encode(data[i+1,:,u_dim:])
            pred_loss += beta * mse_loss(X_next, Y)
        
        # Koopman线性性约束
        X_next_encoded = net.encode(X_next[:,:Nstate])
        lin_loss += mse_loss(X_next_encoded, X_next)
        # 控制重建误差
        u_rec = net.control_decode(hat_u)
        recon_loss += mse_loss(u_rec, data[i,:,:u_dim])

        X_current = X_next
        beta *= gamma
    
    pred_loss = pred_loss / beta_sum if beta_sum > 0 else pred_loss
    lin_loss = lin_loss / beta_sum if beta_sum > 0 else lin_loss
    recon_loss = recon_loss / beta_sum if beta_sum > 0 else recon_loss

    # 总损失 = 预测损失 + 线性性约束 + 流形几何约束
    total_loss = pred_loss + lambda_lin * lin_loss + lambda_geom * geom_loss + lambda_recon * recon_loss
    return total_loss, pred_loss, lin_loss, geom_loss, recon_loss

# 特征值损失
def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
    mask = c > 0
    loss = c[mask].sum()
    return loss

# 训练ODE流形模型
def train_state_manifold(data, input_dim, epochs=1000, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ode_model = ODEManifold(input_dim).to(device).double()
    optimizer = torch.optim.Adam(ode_model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    # 准备训练数据：确保数据张量在正确设备且为double类型
    steps, traj_num, dim = data.shape
    u_dim = dim - input_dim  # 数据格式：[时间步, 轨迹数, 控制维+状态维]
    # 转换为double并移动到设备，避免类型不匹配
    states = torch.DoubleTensor(data[:, :, u_dim:]).to(device)  
    
    # 时间点（均匀采样，与数据时间步对应）
    t = torch.linspace(0, 1, steps, device=device)
    
    print(f"开始训练状态ODE流形模型，输入维度: {input_dim}，设备: {device}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # 随机采样批次（避免显存溢出，限制批次大小）
        idx = random.sample(range(traj_num), min(256, traj_num))
        batch_states = states[:, idx, :]
        
        optimizer.zero_grad()
        ode_loss = 0.0
        
        # 1. ODE拟合损失：预测下一时刻状态
        for i in range(steps - 1):
            x0 = batch_states[i]  # t时刻状态
            # 积分一步（从t[i]到t[i+1]）
            x_pred = ode_model.integrate(x0, t[i:i+2])  
            x_true = batch_states[i+1]  # t+1时刻真实状态
            ode_loss += mse_loss(x_pred, x_true)
        ode_loss = ode_loss / (steps - 1)  # 平均到每个时间步
        
        # 2. 向量场光滑性正则化：修复梯度跟踪问题
        # 重塑为[样本数, 状态维]，并显式开启梯度
        x_sample = batch_states.reshape(-1, input_dim).requires_grad_(True)
        # 计算向量场 F(x)
        dx = ode_model(x_sample)
        # 计算F对x的梯度（光滑性衡量：梯度越大越不光滑）
        grad_dx = torch.autograd.grad(
            outputs=dx.sum(),  # 对所有样本的向量场求和，便于计算梯度
            inputs=x_sample,
            create_graph=False,  # 无需二阶导数，关闭节省计算
            retain_graph=False
        )[0]  # grad_dx形状：[样本数, 状态维]
        # 光滑性损失：梯度的L2范数均值
        smooth_loss = torch.mean(torch.norm(grad_dx, dim=1))
        
        # 总损失：ODE拟合损失 + 光滑性正则化（权重0.1平衡）
        total_loss = ode_loss + 0.1 * smooth_loss
        
        # 反向传播与优化
        total_loss.backward()
        optimizer.step()
        
        # 每50轮打印日志
        if (epoch + 1) % 50 == 0:
            print(f"ODE训练 epoch {epoch+1}/{epochs}, 总损失: {total_loss.item():.6f}, 拟合损失: {ode_loss.item():.6f}, 光滑损失: {smooth_loss.item():.6f}")
    
    return ode_model

def train_control_manifold(data, input_dim, epochs=1000, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备训练数据：确保数据张量在正确设备且为double类型
    steps, traj_num, dim = data.shape
    u_dim = dim - input_dim  # 数据格式：[时间步, 轨迹数, 控制维+状态维]
    ode_model = ODEManifold(u_dim).to(device).double()
    optimizer = torch.optim.Adam(ode_model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    # 转换为double并移动到设备，避免类型不匹配
    controls = torch.DoubleTensor(data[:, :, :u_dim]).to(device)  
    
    # 时间点（均匀采样，与数据时间步对应）
    t = torch.linspace(0, 1, steps, device=device)
    
    print(f"开始训练控制ODE流形模型，输入维度: {u_dim}，设备: {device}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # 随机采样批次（避免显存溢出，限制批次大小）
        idx = random.sample(range(traj_num), min(256, traj_num))
        batch_states = controls[:, idx, :]
        
        optimizer.zero_grad()
        ode_loss = 0.0
        
        # 1. ODE拟合损失：预测下一时刻状态
        for i in range(steps - 1):
            x0 = batch_states[i]  # t时刻状态
            # 积分一步（从t[i]到t[i+1]）
            x_pred = ode_model.integrate(x0, t[i:i+2])  
            x_true = batch_states[i+1]  # t+1时刻真实状态
            ode_loss += mse_loss(x_pred, x_true)
        ode_loss = ode_loss / (steps - 1)  # 平均到每个时间步
        
        # 2. 向量场光滑性正则化：修复梯度跟踪问题
        # 重塑为[样本数, 状态维]，并显式开启梯度
        x_sample = batch_states.reshape(-1, u_dim).requires_grad_(True)
        # 计算向量场 F(x)
        dx = ode_model(x_sample)
        # 计算F对x的梯度（光滑性衡量：梯度越大越不光滑）
        grad_dx = torch.autograd.grad(
            outputs=dx.sum(),  # 对所有样本的向量场求和，便于计算梯度
            inputs=x_sample,
            create_graph=False,  # 无需二阶导数，关闭节省计算
            retain_graph=False
        )[0]  # grad_dx形状：[样本数, 状态维]
        # 光滑性损失：梯度的L2范数均值
        smooth_loss = torch.mean(torch.norm(grad_dx, dim=1))
        
        # 总损失：ODE拟合损失 + 光滑性正则化（权重0.1平衡）
        total_loss = ode_loss + 0.1 * smooth_loss
        
        # 反向传播与优化
        total_loss.backward()
        optimizer.step()
        
        # 每50轮打印日志
        if (epoch + 1) % 50 == 0:
            print(f"ODE训练 epoch {epoch+1}/{epochs}, 总损失: {total_loss.item():.6f}, 拟合损失: {ode_loss.item():.6f}, 光滑损失: {smooth_loss.item():.6f}")
    
    return ode_model

# 主训练函数
def train(env_name, train_steps=15000, suffix="", all_loss=0,
          encode_dim=12, b_dim=2, layer_depth=3, e_loss=1, gamma=0.99,
          detach=0, Ktrain_samples=50000, lambda_geom=0.1, lambda_lin=0.1):
    
    # 数据准备
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    Ktest_data = data_collect.collect_koopman_data(20000, 15, mode="eval")
    print("测试数据准备完成，形状:", Ktest_data.shape)
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples, 15, mode="train")
    print("训练数据准备完成，形状:", Ktrain_data.shape)
    
    Ktrain_samples = Ktrain_data.shape[1]
    in_dim = Ktest_data.shape[-1] - u_dim
    Nstate = in_dim
    
    # 第一步：训练ODE流形模型
    print("开始训练ODE流形模型...")
    # state_manifold = train_state_manifold(Ktrain_data, in_dim)
    control_manifold = train_control_manifold(Ktrain_data, in_dim)
    state_manifold = train_state_manifold(Ktrain_data, in_dim)
    # 网络参数设置
    layer_width = 128
    control_encode_dim = in_dim + u_dim + b_dim
    layers = [in_dim] + [layer_width] * layer_depth + [encode_dim]
    belayers = [in_dim + u_dim] + [layer_width] * layer_depth + [b_dim]
    # bdlayers = [in_dim + u_dim + b_dim] + [layer_width] * layer_depth + [u_dim]
    bdlayers = [control_encode_dim] + [layer_width] * layer_depth + [in_dim + u_dim]
    Nkoopman = in_dim + encode_dim
    print("网络层结构:", layers)
    
    # 初始化模型
    net = Network(layers, belayers, bdlayers, Nkoopman, u_dim, control_encode_dim)
    if torch.cuda.is_available():
        net.cuda() 
    net.double()
    
    # 优化器和损失函数
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    # 训练配置
    eval_step = 1000
    best_loss = 1000.0
    best_state_dict = {}
    
    # 日志和保存路径
    logdir = f"../Data/{suffix}/MCDKN_{env_name}_layer{layer_depth}_edim{encode_dim}_eloss{e_loss}_gamma{gamma}_aloss{all_loss}_detach{detach}_bdim{b_dim}_samples{Ktrain_samples}_geom{lambda_geom}"
    os.makedirs("../Data/" + suffix, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    
    start_time = time.process_time()
    print("开始训练带流形约束的DKN模型...")
    
    for i in range(train_steps):
        # 随机采样批次
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:, Kindex[:100], :]
        
        # 计算带流形约束的损失
        total_loss, pred_loss, lin_loss, geom_loss, recon_loss = Klinear_loss_with_manifold(
            X, net, state_manifold, control_manifold, mse_loss, u_dim, gamma, Nstate, all_loss, 
            detach, lambda_geom, lambda_lin
        )
        
        # 特征值损失
        Eloss = Eig_loss(net) if e_loss else torch.tensor(0.0)
        loss = total_loss + Eloss
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        # 记录训练损失
        writer.add_scalar('Train/total_loss', loss, i)
        writer.add_scalar('Train/pred_loss', pred_loss, i)
        writer.add_scalar('Train/lin_loss', lin_loss, i)
        writer.add_scalar('Train/geom_loss', geom_loss, i)
        writer.add_scalar('Train/recon_loss', recon_loss, i)
        writer.add_scalar('Train/Eloss', Eloss, i)
        
        
        # 评估和保存模型
        if (i + 1) % eval_step == 0:
            with torch.no_grad():
                eval_loss, _, _, _, _ = Klinear_loss_with_manifold(
                    Ktest_data, net, state_manifold, control_manifold, mse_loss, u_dim, gamma, 
                    Nstate, all_loss=0, detach=detach, lambda_geom=0, lambda_lin=0
                )
                eval_loss_val = eval_loss.detach().cpu().numpy().item()
                
                writer.add_scalar('Eval/loss', eval_loss_val, i)
                
                if eval_loss_val < best_loss:
                    best_loss = copy(eval_loss_val)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {
                        'model': best_state_dict, 
                        'layer': layers, 
                        'belayer': belayers,
                        'bdlayer': bdlayers,
                        'state_manifold': state_manifold.state_dict(),
                        'control_manifold': control_manifold.state_dict()
                    }
                    torch.save(Saved_dict, logdir + ".pth")
                
                print(f"步骤: {i}, 评估损失: {eval_loss_val:.6f}, 最佳损失: {best_loss:.6f}")
        
        writer.add_scalar('Eval/best_loss', best_loss, i)
    
    print(f"训练结束，最佳损失: {best_loss}")
    return best_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="DampingPendulum")
    parser.add_argument("--suffix", type=str, default="5_2_ode_constraint")
    parser.add_argument("--all_loss", type=int, default=1)
    parser.add_argument("--e_loss", type=int, default=0)
    parser.add_argument("--K_train_samples", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--encode_dim", type=int, default=20)
    parser.add_argument("--b_dim", type=int, default=1)
    parser.add_argument("--detach", type=int, default=1)
    parser.add_argument("--layer_depth", type=int, default=3)
    parser.add_argument("--lambda_geom", type=float, default=0.1, help="流形几何约束权重")
    parser.add_argument("--lambda_lin", type=float, default=0.1, help="线性性约束权重")
    args = parser.parse_args()
    
    train(args.env, 
          suffix=args.suffix, 
          all_loss=args.all_loss,
          encode_dim=args.encode_dim, 
          layer_depth=args.layer_depth,
          e_loss=args.e_loss, 
          gamma=args.gamma, 
          detach=args.detach,
          b_dim=args.b_dim, 
          Ktrain_samples=args.K_train_samples,
          lambda_geom=args.lambda_geom,
          lambda_lin=args.lambda_lin)

if __name__ == "__main__":
    main()
