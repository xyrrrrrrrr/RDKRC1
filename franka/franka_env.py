import pybullet as pb
import pybullet_data
import numpy as np
from gym import spaces
class FrankaEnv(object):

    def __init__(self, render=True, ts=0.002):

        self.frame_skip=10
        if render:
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)
        pb.setTimeStep(ts)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeID = pb.loadURDF('plane.urdf')
        self.robot = pb.loadURDF('./franka_description/robots/franka_panda.urdf', [0.,0.,0.], useFixedBase=1)
        pb.setGravity(0,0,-9.81)
        self.reset_joint_state = [0., -0.78, 0., -2.35, 0., 1.57, 0.78]
        self.ee_id = 7
        self.sat_val = 0.3
        self.reset()
        self.joint_low = np.array([-2.9,-1.8,-2.9,-3.0,-2.9,-0.08,-2.9])
        self.joint_high = np.array([2.9,1.8,2.9,0.08,2.9,3.0,2.9])
        # 状态空间定义（总维度17，需与self.Nstates一致）
        self.ee_pos_low = [-1.0, -1.0, 0.0]  # 末端位置下界（示例：工作空间范围）
        self.ee_pos_high = [1.0, 1.0, 1.0]   # 末端位置上界（根据机械臂工作空间调整）
        self.joint_vel_low = np.array([-2.175] * 7)  # 关节速度下限（单位：rad/s，参考Franka参数）
        self.joint_vel_high = np.array([2.175] * 7)  # 关节速度上限
        
        # 拼接状态空间上下界（总长度7+7+3=17）
        self.state_low = np.concatenate([
            self.ee_pos_low,       # 末端位置
            self.joint_low,       # 关节位置
            self.joint_vel_low

        ], dtype=np.float32)
        self.state_high = np.concatenate([
            self.ee_pos_high,      # 末端位置
            self.joint_high,     # 关节位置
            self.joint_vel_high
        ], dtype=np.float32)
        
        # 定义观测空间和动作空间
        self.observation_space = spaces.Box(
            low=self.state_low,
            high=self.state_high,
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-self.sat_val, high=self.sat_val, shape=(7,), dtype=np.float32)
        self.Nstates = 17
        self.udim = 7
        self.dt = self.frame_skip*ts
        
    def get_ik(self, position, orientation=None):
        if orientation is None:
            jnts = pb.calculateInverseKinematics(self.robot, self.ee_id, position)[:7]
        else:
            jnts = pb.calculateInverseKinematics(self.robot, self.ee_id, position, orientation)[:7]
        return jnts
        
    def get_state(self):
        jnt_st = pb.getJointStates(self.robot, range(7))
        ## 1,2,3 position 
        ## 4,5,6 are the link frame pose, orientation in quat and ve
        ee_state = pb.getLinkState(self.robot, self.ee_id)[-2:] ### Why local and not the Cartesian ones? Ask Ian 
        jnt_ang = []
        jnt_vel = []
        for jnt in jnt_st:
            jnt_ang.append(jnt[0])
            jnt_vel.append(jnt[1])
        # print(ee_state[0],ee_state[1],jnt_ang,jnt_vel)
        self.state = np.concatenate([ee_state[0], ee_state[1], jnt_ang, jnt_vel]) # ee_state[0] are [x,y,z] of EE, ee_state[1] are quaternions: x,y,z,w of EE
        return self.state.copy()

    def reset(self):
        for i, jnt in enumerate(self.reset_joint_state):
            pb.resetJointState(self.robot, i, self.reset_joint_state[i])
        return self.get_state()

    def reset_state(self,joint):
        for i, jnt in enumerate(joint):
            pb.resetJointState(self.robot, i, joint[i])
        return self.get_state()

    def step(self, action):
        a = np.clip(action, -self.sat_val, self.sat_val)
        pb.setJointMotorControlArray(
                    self.robot, range(7),
                    pb.VELOCITY_CONTROL, targetVelocities=a)
        for _ in range(self.frame_skip):
            pb.stepSimulation()

        return self.get_state()
