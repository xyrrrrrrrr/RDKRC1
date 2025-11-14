import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import scipy
import scipy.linalg
import sys
sys.path.append("../utility")
sys.path.append("../train")
sys.path.append('../')
from Utility import data_collecter
import lqr
import os
import matplotlib.pyplot as plt

data_dir = "./data/"

# 方法与样式映射（颜色、标签）
method_info = {
    "KoopmanRBF": {"color": "orange", "label": "KRBF(Failed)"},
    "KoopmanU": {"color": "green", "label": "DKUC"},
    "KoopmanNonlinearA": {"color": "purple", "label": "DKAC"},
    "DKNGU": {"color": "red", "label": "MCDKN(Ours)"},
}
# 起点、终点标记样式
start_marker = {"marker": "o", "color": "blue", "label": "Start", "markersize": 12}
goal_marker = {"marker": "*", "color": "red", "label": "Goal", "markersize": 12}

data_dir = "./data/"  # 轨迹数据存储目录


def plot_trajectory(env_name, x_dim, y_dim, x_label, y_label, x_start, y_start, x_ref, y_ref):
    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.title(f"{env_name}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.xlim(-5,2)
    plt.xlim(-2,1.0)
    # plt.ylim(-2,16)
    plt.ylim(-0.5,1.0)
    # plt.ylim(-6,18)
    for method, info in method_info.items():
        trajectory_path = os.path.join(data_dir, f"{env_name}_{method}_observations.npy")
        if os.path.exists(trajectory_path):
            trajectory = np.load(trajectory_path)
            # 假设轨迹维度为 (时间步, 状态维度)，
            x = trajectory[x_dim, :]
            y = trajectory[y_dim, :]
            if method == "KoopmanRBF":
                x = x[0]
                y = y[0]
            # if method == "KoopmanNonlinearA":
            #     x = x[0]
            #     y = y[0]
            plt.plot(x, y, color=info["color"], label=info["label"])
        
        else:
            print(f"提示：{trajectory_path} 数据文件不存在，请检查路径和命名。")
    plt.plot(x_start, y_start, **start_marker)
            # 绘制终点（轨迹最后一个点）
    plt.plot(x_ref, y_ref, **goal_marker)
    plt.legend()
    plt.savefig(f"./fig/{env_name}_trajectory.png")
    plt.show()


if __name__ == "__main__":
    envs = ["LunarLanderContinuous-v2","CartPole-v1","Pendulum-v1","DampingPendulum","DoublePendulum"]
    x_dim = [3,1,0,0,0]
    y_dim = [1,3,1,1,2]
    x_label = ["dY", "dX","theta","theta","theta1"]
    y_label = ["Y", "dtheta","dtheta","dtheta","theta2"]
    x_start = [0.15, -1.0, -3.0, -3, -1.5]
    y_start = [1.4, 0.0, 0.5, 2, 0.1]
    x_ref = [0,0,0,0,0]
    y_ref = [0,0,0,0,0]
    env_id = 1
    
    plot_trajectory(envs[env_id],x_dim[env_id],y_dim[env_id],x_label[env_id],y_label[env_id], x_start[env_id], y_start[env_id], x_ref[env_id], y_ref[env_id])
    # 读取./fig中的五张图片，将他们合并到一张图中，按照（1， 2， 2）的排列方式排列
    # from PIL import Image
    # fig_names = ["DampingPendulum_trajectory.png", "Pendulum-v1_trajectory.png","LunarLanderContinuous-v2_trajectory.png","CartPole-v1_trajectory.png","DoublePendulum_trajectory.png"]
    # images = [Image.open(f"./fig/{name}")
    #             for name in fig_names]
    # # 创建一个新的空白图像，用于合并
    # new_image = Image.new('RGB', (800, 1800))
    # # 第一张图占据两张图片的位置，其他四张图各占据一张图片的位置
    # new_image.paste(images[0], (0, 0, 800, 600))
    # images[1] = images[1].resize((400, 300))
    # images[2] = images[2].resize((400, 300))
    # images[3] = images[3].resize((400, 300))
    # images[4] = images[4].resize((400, 300))
    # new_image.paste(images[1], (0, 600, 400, 900))
    # new_image.paste(images[2], (400, 600, 800, 900))
    # new_image.paste(images[3], (0, 900, 400, 1200))
    # new_image.paste(images[4], (400, 900, 800, 1200))
    # # 保存合并后的图像
    # new_image.save('./fig/combined_trajectory.png')

