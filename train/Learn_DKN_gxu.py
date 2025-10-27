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
from torchdiffeq import odeint
# from scipy.integrate import odeint
from Utility import data_collecter
import time
import tqdm

# 定义ODE流形拟合网络
class ODEManifold(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_steps=50):
        super(ODEManifold, self).__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        # ODE向量场参数化网络
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        # 确保向量场在流形上
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, t):
        """计算向量场 F(x, t)"""
        ts = torch.ones([x.shape[0],1], device=x.device) * t
        x = torch.concatenate([x, ts], axis=1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return self.norm(x)  # 归一化确保平滑性
    
    def integrate(self, x0, t):
        """用RK4阶积分ODE, 保证精度"""
        # 定义ODE函数：dx/dt = f(x, t)
        def ode_func(t, x):
            return self.forward(x, t)
        x_pred = odeint(ode_func, x0, t, method='rk4')
        result =  x_pred[-1]  # 返回最后一个时间步的结果
        return result
    
    def compute_pushforward(self, x):
        """批量计算微分同胚φ(x)的雅可比，避免循环，利用GPU并行"""
        x = x.requires_grad_(True)  # 批量开启梯度，无需逐个处理
        t_ode = torch.tensor([0.0, 1.0], device=x.device)
        
        # 批量计算雅可比：torch.autograd.functional.jacobian支持批量输入
        # 输入x: [B, D]，输出雅可比: [B, D, D]（B=batch_size, D=input_dim）
        J = torch.autograd.functional.jacobian(
            func=lambda z: self.integrate(z, t_ode),
            inputs=x,
            create_graph=False,  # 无需二阶导数，关闭节省计算
            vectorize=True  # 开启向量化计算，进一步加速
        )
        # 调整形状：jacobian输出为[B, D, B, D] → 简化为[B, D, D]（批量样本独立）
        J = J.diagonal(dim1=0, dim2=2).permute(0, 2, 1)
        return J

    def geodesic_distance(self, x, y):
        t = torch.linspace(0, 1, self.n_steps, device=x.device)
        path = x.unsqueeze(0) + t.unsqueeze(1).unsqueeze(2) * (y - x).unsqueeze(0)  # [S, B, D]
        batch_size = path.shape[1]
        sample_step = 5  # 每隔5个step采样一次雅可比（可根据精度调整，建议3-10）
        
        # 1. 稀疏采样路径点，计算雅可比
        sample_indices = torch.arange(0, self.n_steps, sample_step, device=x.device)
        sampled_path = path[sample_indices]  # [S_sample, B, D]，S_sample = S//sample_step
        J_sampled = self.compute_pushforward(sampled_path.reshape(-1, self.input_dim))  # [S_sample*B, D, D]
        J_sampled = J_sampled.reshape(len(sample_indices), batch_size, self.input_dim, self.input_dim)  # [S_sample, B, D, D]
        
        # 2. 线性插值补全所有step的雅可比（保持精度的同时减少计算）
        J = torch.nn.functional.interpolate(
            J_sampled.permute(1, 2, 3, 0),  # [B, D, D, S_sample] → 适配插值维度
            size=self.n_steps,
            mode='linear',
            align_corners=True
        ).permute(3, 0, 1, 2)  # 恢复为[S, B, D, D]
        
        # 3. 计算真实速度（优化点2.1）
        velocity = torch.gradient(path, dim=0, spacing=t)[0]  # [S-1, B, D]
        # 裁剪雅可比到S-1个step（与速度维度匹配）
        J = J[:-1]  # [S-1, B, D, D]
        
        # 4. 拉回内积计算（保持文档定义不变）
        velocity = velocity.unsqueeze(-1)  # [S-1, B, D, 1]
        J_T = J.transpose(-2, -1)
        inner_prod = torch.matmul(torch.matmul(torch.matmul(velocity.transpose(-2, -1), J_T), J), velocity)
        step_distances = torch.sqrt(inner_prod.squeeze(-1).squeeze(-1))
        
        # 5. 积分求距离
        dist = torch.trapz(step_distances, t[:-1], dim=0)
        return dist
    
    def reverse_integrate(self, x1, t):
        """反向积分：从x1积分回t=0的点"""
        def ode_func_rev(t, x):
            return -self.forward(x, t)  # 反向时间：dx/dt = -f(x)
        x_rec = odeint(ode_func_rev, x1, t.flip(0))  # 时间反转
        result = x_rec[-1].detach()
        return result

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
        y = torch.cat([u, x], axis=-1) 
        gy = self.control_encoder(y)
        return torch.cat([y, gy], axis=-1)

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
def Klinear_loss_with_manifold(data, net, ode_manifold, mse_loss, u_dim=1, gamma=0.99, 
                               Nstate=4, all_loss=0, detach=0, lambda_geom=0.1, lambda_control=0.1, lambda_recon=0.1):
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
        id_traj = torch.randint(0, train_traj_num, (min(100, batch_size//2),), device=device)
        idx_time = torch.randint(0, steps - 1,  (1,), device=device)
        idy_time = idx_time + 1
        x_samples = data[idx_time, id_traj, :]
        y_samples = data[idy_time, id_traj, :]
        statex_samples = states[idx_time, id_traj, :]
        statey_samples = states[idy_time, id_traj, :]
        ux_samples = controls[idx_time, id_traj, :]
        uy_samples = controls[idy_time, id_traj, :]

        # 计算流形上的测地距离
        geo_dist = ode_manifold.geodesic_distance(x_samples, y_samples)

        # 计算编码空间中的距离
        encoded_statex = net.encode(statex_samples)
        encoded_statey = net.encode(statey_samples)
        encoded_ux = net.control_encode(statex_samples, ux_samples)
        encoded_uy = net.control_encode(statey_samples, uy_samples)
        encoded_x = torch.concatenate([encoded_statex, encoded_ux], dim=1)
        encoded_y = torch.concatenate([encoded_statey, encoded_uy], dim=1)
        encoded_dist = torch.norm(encoded_x - encoded_y, dim=1)

        # 计算比例因子c，使编码距离与测地距离成比例
        c = torch.sum(encoded_dist * geo_dist) / (torch.sum(geo_dist ** 2) + 1e-8) 
        # 几何约束损失
        geom_loss = torch.mean(torch.abs(encoded_dist - c * geo_dist))
    
    # 原始Koopman损失计算
    Z_current = net.encode(data[0,:,u_dim:])
    beta = 1.0
    beta_sum = 0.0
    pred_loss = torch.zeros(1, dtype=torch.float64).to(device)
    lin_loss = torch.zeros(1, dtype=torch.float64).to(device)
    recon_loss = torch.zeros(1, dtype=torch.float64).to(device)
    # Z_current[:,:Nstate] = X_current
    for i in range(steps-1):
        hat_u = net.control_encode(Z_current[:,:Nstate].detach() if detach else Z_current[:,:Nstate], 
                             data[i,:,:u_dim])
        Z_next = net.forward(Z_current, hat_u)
        beta_sum += beta
        Z_next_encoded = net.encode(data[i+1,:,u_dim:])
        # Koopman线性性约束
        if not all_loss:
            pred_loss += beta * mse_loss(Z_next[:,:Nstate], data[i+1,:,u_dim:])
        else:
            pred_loss += beta * mse_loss(Z_next, Z_next_encoded)
        # 重建误差        
        x_rec = Z_current[:,u_dim:] 
        u_rec = hat_u[:,:u_dim]
        recon_loss += mse_loss(u_rec, data[i,:,:u_dim]) + mse_loss(x_rec, data[i:,:,u_dim:])
        Z_current = Z_next
        beta *= gamma
    
    pred_loss = pred_loss / beta_sum if beta_sum > 0 else pred_loss

    control_loss = Eig_loss(net) + Controlability_loss(net)
    control_loss = control_loss / beta_sum if beta_sum > 0 else control_loss

    recon_loss = recon_loss / beta_sum if beta_sum > 0 else recon_loss

    total_loss = pred_loss + lambda_geom * geom_loss + lambda_control * control_loss + lambda_recon * recon_loss

    return total_loss, pred_loss, control_loss, geom_loss, recon_loss

# 特征值损失
def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs() - torch.ones(1, dtype=torch.float64).to(device)
    mask = c > 0
    loss = c[mask].sum()
    return loss

# 能控性损失
def Controlability_loss(net):
    A = net.lA.weight
    B = net.lB.weight
    n = A.size(0)  # 获取状态维度n
    
    # 构建能控性矩阵 C = [B, AB, A²B, ..., A^{n-1}B]
    controllability_matrices = []
    current = B  # 初始项：A^0B = B
    controllability_matrices.append(current)
    
    # 迭代计算 A^k B (k从1到n-1)
    for k in range(1, n):
        current = torch.matmul(A, current)  # A^k B = A·(A^{k-1}B)
        controllability_matrices.append(current)

    # 按列拼接得到能控性矩阵 C ∈ R^{n×(n·m)}
    C = torch.cat(controllability_matrices, dim=1)

    # 计算能控性矩阵的奇异值，最小奇异值反映秩稳健性
    _, S, _ = torch.linalg.svd(C, full_matrices=False)  # S为奇异值向量
    min_singular = S[-1]  # 最小奇异值
    
    varepsilon = 1e-6  # 避免数值不稳定的小常数
    loss = -min_singular + varepsilon  # 当最小奇异值 ≥ epsilon时，损失趋近于0
    
    return loss.clamp(min=0.0)  # 确保损失非负（奇异值过小时才产生惩罚）
    
# 训练ODE流形模型
def train_ode_manifold(data, input_dim, epochs=1000, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ode_model = ODEManifold(input_dim).to(device).double()
    optimizer = torch.optim.Adam(ode_model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    # 准备训练数据：确保数据张量在正确设备且为double类型
    steps, traj_num, dim = data.shape
    u_dim = dim - input_dim  # 数据格式：[时间步, 轨迹数, 控制维+状态维]
    # 转换为double并移动到设备，避免类型不匹配
    states = torch.DoubleTensor(data[:, :, :]) 
    t_ode = torch.tensor([0.0, 1.0]).to(device)
    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        total_loss = 0.0
        ode_loss = 0.0
        recon_loss = 0.0
        # 随机采样批次（避免显存溢出，限制批次大小）
        idx = random.sample(range(traj_num), min(256, traj_num))
        batch_states = states[:, idx, :].to(device) 
        optimizer.zero_grad()
        # idtime = random.sample(range(steps - 1))
        # 1. ODE拟合损失：预测下一时刻状态
        for i in range(steps - 1):
            x0 = batch_states[i]  # t时刻状态
            # 积分一步（从t[i]到t[i+1]）
            x_pred = ode_model.integrate(x0, t_ode) 
            x_true = batch_states[i+1] # t+1时刻真实状态
            x_rec = ode_model.reverse_integrate(x_true, t_ode)
            ode_loss += mse_loss(x_pred[:, u_dim:], x_true[:, u_dim:])
            recon_loss += mse_loss(x0, x_rec)
            del x0, x_pred, x_true, x_rec
        ode_loss = ode_loss / (steps - 1)  # 平均到每个时间步
        recon_loss = recon_loss / (steps - 1)
        # 总损失
        # total_loss = 0.5* ode_loss + 0.5 * recon_loss
        total_loss = ode_loss    
        # 反向传播与优化
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        pbar.set_description_str(f"ODE训练 epoch {epoch+1}/{epochs}, 总损失: {total_loss.item():.6f}, 拟合损失: {ode_loss.item():.6f}, 重建损失: {recon_loss.item():.6f}")
        del batch_states, ode_loss, recon_loss, total_loss
        torch.cuda.empty_cache()  # 释放未引用的显存


    return ode_model



# 主训练函数
def train(env_name, train_steps=200000, suffix="", all_loss=0,
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
    ode_manifold = train_ode_manifold(Ktrain_data, in_dim + u_dim)
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
    
    for i in tqdm.trange(train_steps):
        # 随机采样批次
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:, Kindex[:100], :]
        
        # 计算带流形约束的损失
        total_loss, pred_loss, lin_loss, geom_loss, recon_loss = Klinear_loss_with_manifold(
            X, net, ode_manifold, mse_loss, u_dim, gamma, Nstate, all_loss, 
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
                    Ktest_data, net, ode_manifold, mse_loss, u_dim, gamma, 
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
                        'ode_manifold': ode_manifold.state_dict(),
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
    parser.add_argument("--K_train_samples", type=int, default=200)
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
