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

    def _build_context(self, current_time=None):
        if self.static_environment is None:
            raise ValueError('static_environment 不能为空；DynamicALNS 不在类内部初始化静态矩阵。')

        topology_bundle = self.static_environment.get_topology_bundle()
        traffic_bundle = self.traffic_provider.get_snapshot(current_time=current_time)

        dist_matrix_car = traffic_bundle.get('dist_matrix_car', topology_bundle['dist_matrix_car'])
        dist_matrix_uav = traffic_bundle.get('dist_matrix_uav', topology_bundle['dist_matrix_uav'])
        risk_array = traffic_bundle.get('risk_array', topology_bundle['risk_array'])

        if getattr(self.config, 'uav_time_scale', 1.0) != 1.0:
            dist_matrix_uav = dist_matrix_uav * float(self.config.uav_time_scale)

        if self.hub_selector is not None:
            hub_indices = self.hub_selector(topology_bundle, current_time=current_time)
        else:
            hub_indices = list(traffic_bundle.get('hub_indices', []))
            if not hub_indices:
                raise ValueError('缺少 hub_indices，请通过 traffic_provider 或 hub_selector 提供。')

        num_nodes = len(topology_bundle['coords_array'])
        target_indices = [i for i in range(num_nodes) if i not in hub_indices]

        r_min = risk_array.min()
        r_max = risk_array.max()
        norm = (risk_array - r_min) / (r_max - r_min) if r_max > r_min else np.zeros(num_nodes)
        node_service_time = (
            self.config.service_time_min
            + norm * (self.config.service_time_max - self.config.service_time_min)
        ).astype(float)

        return build_runtime_context(
            dist_car=dist_matrix_car,
            dist_uav=dist_matrix_uav,
            nodes_risk=risk_array,
            hub_indices=hub_indices,
            target_indices=target_indices,
            node_service_time=node_service_time,
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
            current_time=current_time,
            traffic_matrix=traffic_bundle.get('traffic_matrix'),
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

    def solve(self, current_time=None, max_iterations=1000, use_regret=False):
        if ALNS is None:
            raise ImportError('未安装 alns 库，请先安装后再运行 DynamicALNS。')

        rnd = np.random.RandomState(self.random_seed)
        context = self._build_context(current_time=current_time)
        init_sol = self._build_initial_state(context, rnd)

        alns = ALNS(np.random.RandomState(self.random_seed))
        alns.add_destroy_operator(
            lambda state, random_state: destroy_random(
                state,
                random_state,
                context,
                current_time=current_time,
                traffic_matrix=context.traffic_matrix,
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
