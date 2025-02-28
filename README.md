# 自动驾驶车辆轨迹跟踪

## 项目概述
本项目实现了基于模型预测控制（Model Predictive Control, MPC）的车辆控制系统，集成于CARLA自动驾驶模拟环境中。该系统能够实现精确的路径跟踪控制，同时通过调整控制参数可以获得更平滑的横向转向表现。

## 项目参考
- 公式推导：https://zhuanlan.zhihu.com/p/525523586
- 公式代码：https://github.com/gustavomoers/CollisionAvoidance-Carla-DRL-MPC

## Carla演示(目前为初调版本)
![Carla MPC](https://github.com/SavannaBlad/carla_MPC/blob/main/video/demo.mp4)

## 结果演示
![Result for car](https://github.com/SavannaBlad/carla_MPC/blob/main/fig/result.png)

## 系统特点
- 基于模型预测控制（MPC）的车辆轨迹跟踪
- 运动学自行车模型用于车辆状态预测
- 使用CARLA模拟环境进行仿真测试
- 自适应路径规划与跟踪
- 实时性能分析与可视化
- 可调整的控制参数以优化车辆表现

## 环境要求
- Python 3.7+
- CARLA 0.9.14+
- cvxpy
- numpy
- matplotlib
- carla Python API

## 使用方法
1. 启动carla模拟器
   ```bash
   cd path/to/carla
   ./CarlaUE4.exe -windowed -ResX=800 -ResY=600
   ```
2. 运行MPC控制器
   ```bash
   cd PythonAPI/examples/carla_MPC
   python MPCController2.py
   ```

## 任务列表
1. 确定适当的Q、R、F矩阵，以优化车辆在横向控制上的平稳性和响应性能。
2. 探讨将该项目通过Pygame进行可视化呈现，以增强仿真环境的真实感。
3. 引入更多的约束条件，并对MPC求解器的性能进行优化，以提高计算效率。
4. 收集并分析更多车辆的物理参数，以增强模型的准确性和可扩展性。
5. 对现有MPC控制器进行改进，以提升其在复杂场景下的鲁棒性和稳定性。
