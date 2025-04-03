import numpy as np
import math
import pybullet as p
from collections import deque

try:
    from geo_controller import GeoController
except ImportError:
    from .geo_controller import GeoController

class Controller():
    """无人机控制器类，实现圆形轨迹飞行和动态避障功能"""

    def __init__(self,
                 circle_radius,
                 initial_obs,
                 initial_info,
                 buffer_size: int = 100,
                 verbose: bool = False):
        """
        控制器初始化
        
        参数：
            circle_radius: 圆形轨迹半径
            initial_obs: 无人机初始状态观测值 [x,x_dot,y,y_dot,z,z_dot,...]
            initial_info: 场景初始信息字典
            buffer_size: 数据缓冲区大小
            verbose: 是否打印调试信息
        """
        
        # 从配置中读取控制参数
        self.tolerance = initial_info["position_tolerance"]  # 位置容差
        self.k_p = initial_info["k_p"]  # P控制参数
        self.k_d = initial_info["k_d"]  # D控制参数
        self.radius = circle_radius  # 飞行半径
        
        # 环境参数设置
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]  # 控制时间步长
        self.CTRL_FREQ = initial_info["ctrl_freq"]  # 控制频率
        self.initial_obs = initial_obs  # 初始状态
        self.VERBOSE = verbose  # 调试模式
        self.BUFFER_SIZE = buffer_size  # 缓冲区大小

        # 存储场景先验信息
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]  # 门的位置信息
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]  # 障碍物位置信息
        
        # 初始化障碍物列表
        self.obstacles = []
        if 'nominal_obstacles_pos' in initial_info:
            self.obstacles = initial_info['nominal_obstacles_pos']  # 从配置加载障碍物
        print(self.obstacles)
        
        # 避障参数设置
        self.avoidance_distance = 3.0  # 避障触发距离(米)
        self.avoidance_strength = 2.0  # 避障力度系数
        
        # 初始化几何控制器
        self.ctrl = GeoController()
        self.KF = initial_info["quadrotor_kf"]  # 无人机动力系数
        
        # 重置计数器和缓冲区
        self.reset()
        self.interEpisodeReset()
        
        # 执行轨迹规划
        self.planning(initial_info)

    def planning(self, initial_info):
        """轨迹规划主函数，绘制参考圆形轨迹"""
        # 绘制内外两个边界圆
        self.draw_circle(initial_info, self.radius - self.tolerance)  # 内边界
        self.draw_circle(initial_info, self.radius + self.tolerance)  # 外边界

    def draw_circle(self, initial_info, radius):
        """
        在PyBullet中绘制圆形参考轨迹
        
        参数：
            initial_info: 场景信息
            radius: 要绘制的圆的半径
        """
        # 获取初始位置和偏置
        init_pos = np.array([self.initial_obs[0], self.initial_obs[2], self.initial_obs[4]])
        bias_pos = np.array([self.radius, 0.0, 0.0])
        
        # 计算轨迹点
        duration = 4  # 轨迹持续时间
        omega = 2 * np.pi / duration  # 角速度
        dt = 0.01  # 时间步长
        nsample = int(duration / dt)  # 采样点数
        time = dt * np.arange(nsample)  # 时间序列
        
        # 计算圆形轨迹上的点
        ref_x, ref_y, ref_z = [], [], []
        for i in range(nsample):
            t = time[i]
            ref_x.append((radius) * math.cos(omega * t) - bias_pos[0] + init_pos[0])
            ref_y.append((radius) * math.sin(omega * t) - bias_pos[1] + init_pos[1])
            ref_z.append(-bias_pos[2] + init_pos[2])
            
        # 绘制轨迹线
        self.draw_trajectory(initial_info, np.array(ref_x), np.array(ref_y), np.array(ref_z))   

    def draw_trajectory(self, initial_info, ref_x, ref_y, ref_z):
        """
        在PyBullet GUI中绘制轨迹线
        
        参数：
            initial_info: 场景信息
            ref_x: x坐标序列
            ref_y: y坐标序列 
            ref_z: z坐标序列
        """
        # 分段绘制轨迹线
        step = int(ref_x.shape[0]/50)
        for i in range(step, ref_x.shape[0], step):
            p.addUserDebugLine(
                lineFromXYZ=[ref_x[i-step], ref_y[i-step], ref_z[i-step]],
                lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                lineColorRGB=[1, 0, 0],  # 红色轨迹线
                lineWidth=3,
                physicsClientId=initial_info["pyb_client"])
        
        # 绘制最后一段轨迹线
        p.addUserDebugLine(
            lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
            lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            physicsClientId=initial_info["pyb_client"])

    def computeAction(self, obs, target_p, target_v, target_a):
        """
        计算电机转速控制量
        
        参数：
            obs: 当前状态观测值
            target_p: 目标位置
            target_v: 目标速度
            target_a: 目标加速度
            
        返回：
            电机转速控制量
            位置误差
        """
        # 调用几何控制器计算控制量
        rpms, pos_e, _ = self.ctrl.compute_control(
            self.CTRL_TIMESTEP,
            cur_pos=np.array([obs[0], obs[2], obs[4]]),  # 当前位置
            cur_quat=np.array(p.getQuaternionFromEuler([obs[6], obs[7], obs[8]])),  # 当前姿态
            cur_vel=np.array([obs[1], obs[3], obs[5]]),  # 当前速度
            cur_ang_vel=np.array([obs[9], obs[10], obs[11]]),  # 当前角速度
            target_pos=target_p,  # 目标位置
            target_vel=target_v,  # 目标速度
            target_acc=target_a  # 目标加速度
        )
        return self.KF * rpms**2, pos_e  # 返回电机控制量和位置误差

    def getRef(self, time, obs, reward=None, done=None, info=None):
        """
        获取参考轨迹点(含避障逻辑)
        
        参数：
            time: 当前时间
            obs: 当前状态观测值
            reward: 奖励值(可选)
            done: 是否结束标志(可选)
            info: 额外信息(可选)
            
        返回：
            target_p: 目标位置(含避障偏移)
            target_v: 目标速度
            target_a: 目标加速度
        """
        # 计算基础运动参数
        self.desired_speed = self.initial_obs[3]  # 期望速度
        self.omega = self.desired_speed / self.radius  # 角速度
        self.duration = 2 * np.pi / self.omega  # 一圈所需时间
        
        # 获取初始位置和偏置
        init_pos = np.array([self.initial_obs[0], self.initial_obs[2], self.initial_obs[4]])
        bias_pos = np.array([self.radius, 0.0, 0.0])
        
        # 计算标准圆形轨迹
        omega = self.omega
        omega2 = omega * omega
        nominal_pos = np.array([
            self.radius * math.cos(omega * time),  # x坐标
            self.radius * math.sin(omega * time),  # y坐标
            0.0  # z坐标
        ]) - bias_pos + init_pos  # 标准位置
        
        nominal_vel = np.array([
            -self.radius * omega * math.sin(omega * time),  # x速度
            self.radius * omega * math.cos(omega * time),  # y速度
            0.0  # z速度
        ])  # 标准速度
        
        nominal_acc = np.array([
            -self.radius * omega2 * math.cos(omega * time),  # x加速度
            -self.radius * omega2 * math.sin(omega * time),  # y加速度
            0.0  # z加速度
        ])  # 标准加速度
        
        # 获取当前位置
        current_pos = np.array([obs[0], obs[2], obs[4]])
        
        # 初始化避障向量
        avoidance_vector = np.zeros(3)
        
        # 障碍物检测与避障向量计算
        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle[:3])  # 障碍物位置
            dist = np.linalg.norm(current_pos - obstacle_pos)  # 计算距离
            
            # 如果距离小于避障触发距离
            if dist < self.avoidance_distance:
                # 计算排斥方向(单位向量)
                direction = (current_pos - obstacle_pos) / (dist + 1e-6)  # 加上小量防止除以0
                # 计算排斥力度(与距离成反比)
                strength = self.avoidance_distance / (dist + 1e-6)
                # 累加避障向量
                avoidance_vector += direction * strength * self.avoidance_strength
                print("!!!!!!!!!Collision Warning!!!!!!!!!!")
                print("!!!!!!!!!Collision Warning!!!!!!!!!!")
                print("!!!!!!!!!Collision Warning!!!!!!!!!!")
        
        # 合成最终目标位置(标准位置+避障偏移)
        target_p = nominal_pos + avoidance_vector
        target_v = nominal_vel  # 保持原速度
        target_a = nominal_acc  # 保持原加速度
        
        return target_p, target_v, target_a

    def reset(self):
        """重置控制器内部状态和缓冲区"""
        # 初始化数据缓冲区
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)  # 动作缓冲区
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)  # 状态缓冲区
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)  # 奖励缓冲区
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)  # 结束标志缓冲区
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)  # 信息缓冲区
        
        # 重置计数器
        self.interstep_counter = 0  # 步间计数器
        self.interepisode_counter = 0  # 回合间计数器

    def interEpisodeReset(self):
        """重置学习相关计时变量"""
        self.interstep_learning_time = 0  # 步间学习时间
        self.interstep_learning_occurrences = 0  # 步间学习次数
        self.interepisode_learning_time = 0  # 回合间学习时间