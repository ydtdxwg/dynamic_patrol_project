import numpy as np

try:
    from alns import ALNS
    from alns.accept import SimulatedAnnealing
    from alns.select import RouletteWheel
    from alns.stop import MaxIterations
except ImportError:
    ALNS = None
    SimulatedAnnealing = None
    RouletteWheel = None
    MaxIterations = None

from .operators import (
    PatrolState,
    build_runtime_context,
    destroy_random,
    destroy_shaw,
    destroy_worst,
    repair_greedy,
    repair_regret,
)


class DynamicALNS:
    def __init__(self, traffic_provider, config, static_environment=None, hub_selector=None, random_seed=42):
        if traffic_provider is None:
            raise ValueError('traffic_provider 不能为空。')
        self.traffic_provider = traffic_provider
        self.config = config
        self.static_environment = static_environment
        self.hub_selector = hub_selector
        self.random_seed = random_seed

    def _build_context(self, current_time=None, target_indices=None):
        if self.static_environment is None:
            raise ValueError('static_environment 不能为空；DynamicALNS 不在类内部初始化静态矩阵。')

        topology_bundle = self.static_environment.get_topology_bundle()

        # 只使用静态环境中的 UAV 距离矩阵，因为无人机不受地面拥堵影响
        dist_matrix_uav = topology_bundle['dist_matrix_uav']

        if getattr(self.config, 'uav_time_scale', 1.0) != 1.0:
            dist_matrix_uav = dist_matrix_uav * float(self.config.uav_time_scale)

        if self.hub_selector is not None:
            hub_indices = self.hub_selector(topology_bundle, current_time=current_time)
        else:
            # 尝试从静态环境中获取 hub_indices
            hub_indices = list(topology_bundle.get('hub_indices', []))
            if not hub_indices:
                raise ValueError('缺少 hub_indices，请通过 hub_selector 提供。')

        if target_indices is None:
            num_nodes = len(topology_bundle['coords_array'])
            target_indices = [i for i in range(num_nodes) if i not in hub_indices]

        return build_runtime_context(
            dist_uav=dist_matrix_uav,
            hub_indices=hub_indices,
            target_indices=target_indices,
            w_risk=self.config.w_risk,
            w_cover=self.config.w_cover,
            t_max=self.config.t_max,
            uav_endurance=self.config.uav_endurance,
            max_uav_trips=self.config.max_uav_trips,
            num_cars=self.config.num_cars,
            num_uavs=self.config.num_uavs,
            shaw_phi=self.config.shaw_phi,
            shaw_chi=self.config.shaw_chi,
            shaw_noise=self.config.shaw_noise,
            uav_max_stops_per_trip=getattr(self.config, 'uav_max_stops_per_trip', 4),
            traffic_provider=self.traffic_provider,
            current_time_minutes=current_time,
            service_time_min=self.config.service_time_min,
            service_time_max=self.config.service_time_max,
        )

    def _build_initial_state(self, context, rnd):
        init_routes = [[h, h] for h in context.hub_indices]
        init_sol = PatrolState(init_routes, [], list(context.target_indices), context)
        return repair_greedy(
            init_sol,
            rnd,
            context,
            current_time=context.current_time,
            traffic_matrix=context.traffic_matrix,
        )

    def solve(self, current_time=None, max_iterations=1000, use_regret=False, target_indices=None):
        if ALNS is None:
            raise ImportError('未安装 alns 库，请先安装后再运行 DynamicALNS。')

        rnd = np.random.RandomState(self.random_seed)
        context = self._build_context(current_time=current_time, target_indices=target_indices)
        init_sol = self._build_initial_state(context, rnd)

        alns = ALNS(np.random.RandomState(self.random_seed))
        alns.add_destroy_operator(
            lambda state, random_state: destroy_random(
                state,
                random_state,
                context,
                current_time=current_time,
                traffic_matrix=context.traffic_matrix,
                random_state=random_state,
            )
        )
        alns.add_destroy_operator(
            lambda state, random_state: destroy_worst(
                state,
                random_state,
                context,
                current_time=current_time,
                traffic_matrix=context.traffic_matrix,
            )
        )
        alns.add_destroy_operator(
            lambda state, random_state: destroy_shaw(
                state,
                random_state,
                context,
                current_time=current_time,
                traffic_matrix=context.traffic_matrix,
            )
        )
        alns.add_repair_operator(
            lambda state, random_state: repair_greedy(
                state,
                random_state,
                context,
                current_time=current_time,
                traffic_matrix=context.traffic_matrix,
            )
        )
        if use_regret:
            alns.add_repair_operator(
                lambda state, random_state: repair_regret(
                    state,
                    random_state,
                    context,
                    current_time=current_time,
                    traffic_matrix=context.traffic_matrix,
                )
            )

        select = RouletteWheel([2, 2, 8, 0.5], 0.8, 3, 1)
        sa = SimulatedAnnealing(1000, 1, 0.995)
        stop = MaxIterations(max_iterations)
        result = alns.iterate(init_sol, select, sa, stop)

        return {
            'result': result,
            'best_state': result.best_state,
            'best_score': -result.best_state.objective(),
            'context': context,
            'current_time': current_time,
            'destroy_weights': list(select.destroy_weights),
            'repair_weights': list(select.repair_weights),
        }
