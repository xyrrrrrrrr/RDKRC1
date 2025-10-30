from ntpath import join
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import sys
import os
sys.path.append("../utility/")
sys.path.append("../franka/")
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
# physics engine
import pybullet as pb
import pybullet_data
from scipy.io import loadmat, savemat
# Franka simulator
from franka_env import FrankaEnv

#data collect
def Obs(o):
    return np.concatenate((o[:3],o[7:]),axis=0)

class data_collecter():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        self.env =  FrankaEnv(render = False)
        self.Nstates = 17
        self.uval = 0.12
        self.udim = 7
        self.reset_joint_state = np.array(self.env.reset_joint_state)

    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            noise = (np.random.rand(7)-0.5)*2*0.2
            joint_init = self.reset_joint_state+noise
            joint_init = np.clip(joint_init,self.env.joint_low,self.env.joint_high)
            s0 = self.env.reset_state(joint_init)
            s0 = Obs(s0)
            u10 = (np.random.rand(7)-0.5)*2*self.uval
            train_data[0,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                s0 = self.env.step(u10)
                s0 = Obs(s0)
                u10 = (np.random.rand(7)-0.5)*2*self.uval
                train_data[i,traj_i,:]=np.concatenate([u10.reshape(-1),s0.reshape(-1)],axis=0).reshape(-1)
        return train_data

class ManifoldEmbLoss(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.k = k  # K近邻数量
        self.neighbor_indices = None  # 不再预存全局索引，改为batch内临时存储

    def compute_knn(self, X):
        """针对单个batch的X，计算每个样本的K近邻索引（仅在当前batch内）"""
        # 计算X的 pairwise 距离（欧氏距离）
        n = X.shape[0]
        dist_matrix = torch.cdist(X, X, p=2)  # shape=[n, n]
        # 取每个样本的前k+1个近邻（排除自身，所以k+1），再去掉第0个（自身）
        _, indices = torch.topk(dist_matrix, k=self.k+1, largest=False, dim=1)
        self.neighbor_indices = indices[:, 1:]  # shape=[n, k]，每个样本的k个邻居索引
        return self.neighbor_indices

    def forward(self, z, X):
        """
        z: 当前batch的嵌入张量，shape=[batch*T, manifold_dim]
        X: 当前batch的原状态张量，shape=[batch*T, x_dim]
        """
        # 第一步：针对当前batch的X，动态计算K近邻索引
        self.compute_knn(X)
        # 第二步：根据邻居索引，提取z和X的邻居样本
        n = z.shape[0]
        # 确保索引在合法范围内（双重保险）
        self.neighbor_indices = torch.clamp(self.neighbor_indices, 0, n-1)
        
        # 提取每个样本的邻居（shape=[n, k, dim]）
        z_neighbors = z[self.neighbor_indices]  # [n, k, manifold_dim]
        x_neighbors = X[self.neighbor_indices]  # [n, k, x_dim]
        
        # 计算原状态与邻居的距离、嵌入后与邻居的距离
        x_dist = torch.cdist(X.unsqueeze(1), x_neighbors, p=2).squeeze(1) 
        z_dist = torch.cdist(z.unsqueeze(1), z_neighbors, p=2).squeeze(1) 

        x_dist_max = torch.max(x_dist, dim=1, keepdim=True)[0]
        x_dist_max = torch.clamp(x_dist_max, min=1e-8)  # 防止过小导致梯度爆炸
        x_dist = x_dist / x_dist_max  # 归一化，避免尺度
        z_dist_max = torch.max(z_dist, dim=1, keepdim=True)[0]
        z_dist_max = torch.clamp(z_dist_max, min=1e-8)  # 防止过小导致梯度爆炸
        z_dist = z_dist / z_dist_max  # 归一化，避免尺度
        
        # # 计算几何一致性损失
        loss = torch.mean(torch.abs(z_dist - x_dist))

        return loss

#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self, state_encode_layers, control_encode_layers, Nkoopman, u_dim, control_output_dim):
        super(Network, self).__init__()
        Statelayers = OrderedDict()
        for layer_i in range(len(state_encode_layers)-1):
            Statelayers["linear_{}".format(layer_i)] = nn.Linear(state_encode_layers[layer_i], state_encode_layers[layer_i+1])
            if layer_i != len(state_encode_layers)-2:
                Statelayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.state_encoder = nn.Sequential(Statelayers)
        
        Controllayers = OrderedDict()
        for layer_i in range(len(control_encode_layers)-1):
            Controllayers["linear_{}".format(layer_i)] = nn.Linear(control_encode_layers[layer_i], control_encode_layers[layer_i+1])
            if layer_i != len(control_encode_layers)-2:
                Controllayers["relu_{}".format(layer_i)] = nn.ReLU()
        self.control_encoder = nn.Sequential(Controllayers)  
    
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        # self.lB = nn.Linear(belayers[-1], Nkoopman, bias=False)
        self.lB = nn.Linear(control_output_dim, Nkoopman, bias=False)
    # 状态提升
    def encode(self, x):
        return torch.cat([x, self.state_encoder(x)], axis=-1)
    
    # 控制编码
    def control_encode(self, gx, u):
        y = torch.cat([u, gx], axis=-1) 
        gy = self.control_encoder(y)
        return torch.cat([y, gy], axis=-1)

    def forward(self, z, hat_u):
        return self.lA(z) + self.lB(hat_u)

#loss function
def Klinear_loss_with_manifold(data, net, mse_loss, emb_loss, u_dim=1, gamma=0.99, 
                               Nstate=4, all_loss=0, detach=0, lambda_geom=0.1, lambda_control=0.1, lambda_recon=0.3):
    steps, train_traj_num, NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    
    # 提取状态数据用于流形约束
    states = data[:, :, u_dim:]
    controls = data[:, :, :u_dim]
    batch_size = states.shape[1]
    
    # 计算流形几何约束损失
    geom_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    if lambda_geom > 0:
        # 随机选择一些点对计算距离约束
        id_traj = torch.randint(0, train_traj_num, (min(100, batch_size//2),), device=device)
        idx_time = torch.randint(0, steps - 1,  (1,), device=device)
        # x_samples = data[idx_time, id_traj, :]
        # y_samples = data[idy_time, id_traj, :]
        statex_samples = states[idx_time, id_traj, :]
        # statey_samples = states[idy_time, id_traj, :]
        # ux_samples = controls[idx_time, id_traj, :]
        # uy_samples = controls[idy_time, id_traj, :]
        # compute z
        embedx_samples = net.encode(statex_samples)
        # embedy_samples = net.encode(statey_samples)
        # get loss
        geom_loss = emb_loss(embedx_samples, statex_samples)  
    # 原始Koopman损失计算
    Z_current = net.encode(data[0,:,u_dim:])
    beta = 1.0
    beta_sum = 0.0
    pred_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    control_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    recon_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    # Z_current[:,:Nstate] = X_current
    for i in range(steps-1):
        hat_u = net.control_encode(Z_current if detach else Z_current, 
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
        x_rec = Z_current[:,:Nstate] 
        u_rec = hat_u[:,:u_dim]
        # recon_loss += (mse_loss(u_rec, data[i,:,:u_dim]) + mse_loss(x_rec, data[i:,:,u_dim:]))
        recon_loss += mse_loss(u_rec, data[i,:,:u_dim])
        Z_current = Z_next
        beta *= gamma
    
    pred_loss = pred_loss / beta_sum if beta_sum > 0 else pred_loss

    if lambda_control > 0:
        control_loss = Eig_loss(net) + Controlability_loss(net)
        control_loss = control_loss / beta_sum if beta_sum > 0 else control_loss

    if lambda_recon > 0:
        recon_loss = recon_loss / beta_sum if beta_sum > 0 else recon_loss

    total_loss = pred_loss + lambda_geom * geom_loss + lambda_control * control_loss + lambda_recon * recon_loss

    return total_loss, pred_loss, control_loss, geom_loss, recon_loss

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

def Stable_loss(net,Nstate):
    x_ref = np.zeros(Nstate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ref_lift = net.encode_only(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(x_ref_lift)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

def train(env_name,train_steps = 200000,suffix="",all_loss=0,\
            encode_dim = 20,layer_depth=3,e_loss=1,gamma=0.5):
    np.random.seed(98)
    # Ktrain_samples = 1000
    # Ktest_samples = 1000
    Ktrain_samples = 50000
    Ktest_samples = 20000
    Ksteps = 10
    Kbatch_size = 512
    u_dim = 7
    #data prepare
    data_collect = data_collecter(env_name)
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Ksteps)
    print("test data ok!")
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ksteps)
    print("train data ok!")
    # savemat('FrankaTrainingData.mat',{'Train_data':Ktrain_data,'Test_data':Ktest_data})
    # raise NotImplementedError
    # 网络参数设置
    in_dim = Ktest_data.shape[-1]-u_dim
    Nstate = in_dim
    b_dim = encode_dim
    layer_width = 128
    state_input_dim = in_dim
    state_output_dim = in_dim + encode_dim
    control_input_dim = u_dim + state_output_dim
    control_output_dim = control_input_dim + b_dim
    state_encode_layers = [state_input_dim] + [layer_width] * layer_depth + [encode_dim]
    control_encode_layers = [control_input_dim] + [layer_width] * layer_depth + [b_dim]
    Nkoopman = state_output_dim
    net = Network(state_encode_layers, control_encode_layers, Nkoopman, u_dim, control_output_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
    net.double()
    mse_loss = nn.MSELoss()
    emb_loss = ManifoldEmbLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    #train
    eval_step = 1000
    best_loss = 1000.0
    best_state_dict = {}
    subsuffix = suffix+"KK_"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}".format(layer_depth,encode_dim,e_loss,gamma,all_loss)
    logdir = "Data/"+suffix+"/"+subsuffix
    if not os.path.exists( "Data/"+suffix):
        os.makedirs( "Data/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)
    for i in range(train_steps):
        #K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:,Kindex[:Kbatch_size],:]
        total_loss, pred_loss, control_loss, geom_loss, recon_loss = Klinear_loss_with_manifold(
            X, net, mse_loss, emb_loss, u_dim, gamma, Nstate, 1
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step() 
        writer.add_scalar('Train/pred_loss',pred_loss,i)
        writer.add_scalar('Train/control_loss',control_loss,i)
        writer.add_scalar('Train/recon_loss',recon_loss,i)
        writer.add_scalar('Train/geom_loss',geom_loss,i)
        writer.add_scalar('Train/total_loss', total_loss, i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            #K loss
            total_loss, K_loss, control_loss, geom_loss, recon_loss = Klinear_loss_with_manifold(Ktest_data,net, mse_loss, emb_loss, u_dim,gamma,Nstate,all_loss,lambda_control=0,lambda_geom=0,lambda_recon=0)
            Eloss = Eig_loss(net)
            loss = Kloss+Eloss if e_loss else Kloss
            Kloss = Kloss.detach().cpu().numpy()
            Eloss = Eloss.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()
            writer.add_scalar('Eval/Kloss',Kloss,i)
            writer.add_scalar('Eval/Eloss',Eloss,i)
            writer.add_scalar('Eval/loss',loss,i)
            if loss<best_loss:
                best_loss = copy(Kloss)
                best_state_dict = copy(net.state_dict())
                Saved_dict = {'model':best_state_dict,'state_encode_layers': state_encode_layers, 'control_encode_layers': control_encode_layers}
                torch.save(Saved_dict,"Data/"+subsuffix+".pth")
            print("Step:{} Eval-loss{} K-loss:{} E-loss:{}".format(i,loss,Kloss,Eloss))
            # print("-------------END-------------")
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
            e_loss=args.eloss,gamma=args.gamma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="Franka")
    parser.add_argument("--suffix",type=str,default="")
    parser.add_argument("--all_loss",type=int,default=1)
    parser.add_argument("--eloss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.8)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

