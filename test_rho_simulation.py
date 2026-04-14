import numpy as np

# 导入我们的核心架构组件
from data_layer.time_dependent_provider import TimeDependentTrafficProvider
from algorithm.alns_dynamic import DynamicALNS
from algorithm.rho_controller import RHOController

# ==========================================
# 1. 构建模拟环境 (用于独立测试)
# ==========================================
class MockConfig:
    """模拟传入的调度参数配置"""
    def __init__(self):
        self.service_time_min = 3.0   # 最小服务时间 (分钟)
        self.service_time_max = 10.0  # 最大服务时间 (分钟)
        self.w_risk = 1.0             # 风险权重
        self.w_cover = 2.0            # 覆盖权重
        self.t_max = 240.0            # 单车最大巡逻时长 (4小时)
        self.uav_endurance = 40.0     # 无人机续航 (分钟)
        self.max_uav_trips = 4        # 无人机最大架次
        self.num_cars = 3             # 警车数量
        self.num_uavs = 2             # 无人机数量
        self.shaw_phi = 0.5
        self.shaw_chi = 0.5
        self.shaw_noise = 0.1
        self.uav_max_stops_per_trip = 4
        self.uav_time_scale = 1.0

class MockStaticEnv:
    """模拟静态环境，提供无人机距离和初始Hub"""
    def __init__(self, num_nodes):
        self.coords_array = np.random.rand(num_nodes, 2) * 100
        # 无人机距离用简单的欧氏距离模拟
        self.dist_matrix_uav = np.linalg.norm(
            self.coords_array[:, np.newaxis] - self.coords_array[np.newaxis, :], axis=2
        ) * 0.1 
        self.hub_indices = [0] # 假设 0 号节点是中队基站

    def get_topology_bundle(self):
        return {
            'coords_array': self.coords_array,
            'dist_matrix_uav': self.dist_matrix_uav,
            'hub_indices': self.hub_indices
        }

# ==========================================
# 2. 仿真主流程
# ==========================================
def main():
    print("🚀 [1/4] 系统启动：加载四维时空张量数据...")
    # 注意：如果你的 npz 路径不同，AI 请协助用户调整此处路径
    try:
        traffic_provider = TimeDependentTrafficProvider('data/dynamic_map_tensor.npz')
        num_nodes = traffic_provider.risk_tensor.shape[1]
        print(f"✅ 时空张量加载成功！路网节点数: {num_nodes}")
    except Exception as e:
        print(f"❌ 加载张量失败 (请检查路径): {e}")
        return

    # 初始化配置与环境
    config = MockConfig()
    static_env = MockStaticEnv(num_nodes=num_nodes)
    
    # 初始化大脑与总调度
    alns_engine = DynamicALNS(traffic_provider, config, static_env)
    rho_controller = RHOController(alns_engine, traffic_provider)

    # ------------------------------------------
    # 阶段一：早高峰初始全局规划 (t = 84, 即 07:00)
    # ------------------------------------------
    start_time = 84.0 * 5  # 假设时间戳单位为分钟 (07:00 = 420 分钟)
    print(f"\n🌅 [2/4] 阶段一：开始早高峰全局调度规划 (基准时间 t={start_time} min) ...")
    
    # 限制迭代次数以加快测试速度
    initial_result = alns_engine.solve(current_time=start_time, max_iterations=200, use_regret=True)
    state_A = initial_result['best_state']
    
    print(f"✅ 初始规划完成！")
    print(f"📊 捕获总风险与覆盖分: {-state_A.objective():.2f}")
    for i, route in enumerate(state_A.car_routes):
        print(f"   🚓 警车 {i+1} 路径长度: {len(route)} 个节点, 预计耗时: {state_A.car_durations[i]:.1f} 分钟")

    # ------------------------------------------
    # 阶段二：模拟突发事件触发 RHO 截断
    # ------------------------------------------
    # 假设在 08:15 (495 分钟) 发生突发拥堵，需要重调度
    event_time = 495.0 
    print(f"\n🚨 [3/4] 阶段二：突发事件触发！当前时间推进至 t={event_time} min")
    print(f"正在冻结时空，截断已执行的历史轨迹...")
    
    # 偷看 RHO 截取出多少未访问节点
    new_hubs, pending = rho_controller.truncate_state(state_A, event_time)
    print(f"🔍 截断分析: 警车新起点(Hub)被设定为: {new_hubs}")
    print(f"🔍 截断分析: 提取出 {len(pending)} 个尚未访问的目标节点")

    # ------------------------------------------
    # 阶段三：RHO 热启动重新规划
    # ------------------------------------------
    print(f"\n🔥 [4/4] 阶段三：开始热启动重调度...")
    state_B = rho_controller.handle_event_and_reschedule(state_A, event_time=event_time)
    
    print(f"✅ 突发重调度完成！")
    print(f"📊 新方案捕获总风险与覆盖分: {-state_B.objective():.2f}")
    for i, route in enumerate(state_B.car_routes):
        print(f"   🚓 警车 {i+1} 从截断点 {route[0]} 出发 -> 继续执行 {len(route)-1} 个节点任务")

    print("\n🎉 全链路动态 RHO 仿真测试圆满结束！")

if __name__ == '__main__':
    main()
