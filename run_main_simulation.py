from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithm.alns_dynamic import DynamicALNS
from algorithm.rho_controller import RHOController
from data_layer.time_dependent_provider import TimeDependentTrafficProvider


REAL_SQUADS = {
    'Xiangfu': (120.115, 30.315),
    'Gongchen': (120.145, 30.320),
    'Kangqiao': (120.150, 30.360),
    'Daguan': (120.155, 30.300),
    'Banshan': (120.180, 30.350),
    'Shiqiao': (120.190, 30.330),
    'Chengqu': (120.270, 30.170),
    'Chengxiang': (120.250, 30.160),
    'Beigan': (120.265, 30.190),
    'Ningwei': (120.280, 30.220),
    'Xinjie': (120.320, 30.180),
    'Xintang': (120.280, 30.140),
}


class RealPatrolConfig:
    def __init__(self):
        self.service_time_min = 3.0
        self.service_time_max = 10.0
        self.t_max = 240.0
        self.num_cars = 12
        self.num_uavs = 12
        self.uav_endurance = 40.0
        self.max_uav_trips = 6
        self.uav_max_stops_per_trip = 6
        self.uav_time_scale = 1.0
        self.w_risk = 1.0
        self.w_cover = 2.0
        self.shaw_phi = 0.5
        self.shaw_chi = 0.5
        self.shaw_noise = 0.1


class RealStaticEnv:
    def __init__(self, provider: TimeDependentTrafficProvider):
        self.provider = provider
        self.coords_array = provider.coords_array
        self.dist_matrix_uav = provider.dist_matrix_uav
        self.hub_indices, self.hub_info_list = self._setup_real_hubs()
        # TODO: 如 alns-optimization 项目中的真实 index 已最终确认，可直接替换为固定列表。

    def _setup_real_hubs(self):
        hub_indices = []
        hub_info_list = []

        for name, (lon, lat) in REAL_SQUADS.items():
            dists = np.sqrt((self.coords_array[:, 0] - lon) ** 2 + (self.coords_array[:, 1] - lat) ** 2)
            idx = int(np.argmin(dists))
            hub_indices.append(idx)
            hub_info_list.append({'name': name, 'idx': idx, 'coord': (lon, lat)})

        return hub_indices, hub_info_list

    def get_topology_bundle(self):
        return {
            'coords_array': self.coords_array,
            'dist_matrix_uav': self.dist_matrix_uav,
            'hub_indices': self.hub_indices,
        }


def log_step(message: str):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] {message}')


def resolve_tensor_path() -> Path:
    project_dir = Path(__file__).resolve().parent
    return project_dir.parent / 'data' / 'dynamic_map_tensor.npz'


def resolve_result_path() -> Path:
    project_dir = Path(__file__).resolve().parent
    return project_dir.parent / 'data' / 'rho_simulation_results.pkl'


def summarize_hubs(env: RealStaticEnv):
    log_step('中队基站映射完成：以下为基于 alns_turbo.py 真实坐标匹配得到的 hub index')
    for item in env.hub_info_list:
        name = item['name']
        idx = item['idx']
        lon, lat = item['coord']
        print(f'   🏢 {name:<10s} -> 节点索引 {idx:<4d} | 坐标=({lon:.3f}, {lat:.3f})')


def build_valid_target_indices(provider: TimeDependentTrafficProvider, env: RealStaticEnv, start_time_min: float) -> list[int]:
    time_slot_idx = int(start_time_min // provider.TIME_SLOT_MINUTES)
    car_time_slice = provider.car_time_tensor[time_slot_idx]
    num_nodes = provider.risk_tensor.shape[1]
    hub_set = set(env.hub_indices)

    log_step(f'阶段一准备：使用时间片 {time_slot_idx} 执行连通性清洗')

    valid_nodes = []
    for node_idx in range(num_nodes):
        if node_idx in hub_set:
            continue

        min_outbound = float(np.min(car_time_slice[env.hub_indices, node_idx]))
        min_inbound = float(np.min(car_time_slice[node_idx, env.hub_indices]))

        if (
            np.isfinite(min_outbound)
            and np.isfinite(min_inbound)
            and min_outbound < 99999
            and min_inbound < 99999
        ):
            valid_nodes.append(node_idx)

    log_step(f'阶段一准备：清洗后保留 {len(valid_nodes)} 个与基站双向连通的有效节点')

    ranked_nodes = sorted(
        valid_nodes,
        key=lambda idx: provider.get_node_risk(start_time_min, idx),
        reverse=True,
    )
    target_indices = ranked_nodes[:150]

    log_step(f'阶段一准备：按 07:30 风险值提取前 {len(target_indices)} 个高危目标节点')
    return target_indices


def summarize_state(title: str, state):
    total_reward = -state.objective()
    log_step(f'{title} 总收益: {total_reward:.2f}')

    for i, route in enumerate(state.car_routes):
        task_count = max(0, len(route) - 2)
        duration = state.car_durations[i] if i < len(state.car_durations) else 0.0
        print(
            f'   🚓 警车 {i + 1:02d} | 路径节点数={len(route)} | '
            f'任务数={task_count} | 预计耗时={duration:.1f} 分钟'
        )

    print(f'   🛸 无人机任务链数量={len(state.uav_trips)}')
    print(f'   📦 未分配节点数={len(state.unassigned)}')


def run_simulation():
    log_step('系统启动：准备加载动态时空张量数据')
    tensor_path = resolve_tensor_path()
    if not tensor_path.exists():
        raise FileNotFoundError(f'未找到张量文件: {tensor_path}')

    provider = TimeDependentTrafficProvider(str(tensor_path))
    config = RealPatrolConfig()
    env = RealStaticEnv(provider)
    alns_engine = DynamicALNS(provider, config, env)
    rho_controller = RHOController(alns_engine, provider)

    log_step('系统加载完成：Config / StaticEnv / DynamicALNS / RHOController 已初始化')
    summarize_hubs(env)

    # 阶段一：07:30 初始全局规划
    start_time_min = 450.0
    log_step('阶段一开始：07:30 初始全局规划')
    target_indices = build_valid_target_indices(provider, env, start_time_min)

    initial_result = alns_engine.solve(
        current_time=start_time_min,
        max_iterations=1000,
        use_regret=True,
        target_indices=target_indices,
    )
    state_A = initial_result['best_state']
    summarize_state('阶段一完成：初始规划', state_A)

    # 阶段二：08:45 突发事件 RHO 重调度
    event_time_min = 525.0
    log_step('阶段二开始：08:45 突发事件触发 RHO 重调度')

    _, pending_before = rho_controller.truncate_state(state_A, event_time_min)
    log_step(f'阶段二分析：截断后剩余任务节点数 = {len(pending_before)}')

    state_B = rho_controller.handle_event_and_reschedule(state_A, event_time=event_time_min)
    summarize_state('阶段二完成：热启动重调度', state_B)

    reward_A = -state_A.objective()
    reward_B = -state_B.objective()
    log_step(f'阶段二对比：新方案收益变化 = {reward_B - reward_A:.2f}')

    # 阶段三：数据持久化
    result_path = resolve_result_path()
    payload = {
        'state_A': state_A,
        'state_B': state_B,
        'event_time': event_time_min,
        'start_time': start_time_min,
        'target_indices': target_indices,
        'hub_indices': env.hub_indices,
        'hub_info_list': env.hub_info_list,
        'initial_reward': reward_A,
        'rescheduled_reward': reward_B,
        'pending_before_reschedule': list(pending_before),
        'config': config.__dict__.copy(),
    }

    log_step(f'阶段三开始：写入仿真结果到 {result_path}')
    with open(result_path, 'wb') as f:
        pickle.dump(payload, f)
    log_step('阶段三完成：结果持久化成功，完整 RHO 生命周期仿真结束')


if __name__ == '__main__':
    run_simulation()
