from dataclasses import dataclass

import numpy as np

try:
    from alns import State
except ImportError:
    class State:
        pass


@dataclass
class RuntimeContext:
    dist_uav: np.ndarray
    hub_indices: list
    target_indices: list
    w_risk: float
    w_cover: float
    t_max: float
    uav_endurance: float
    max_uav_trips: int
    num_cars: int
    num_uavs: int
    shaw_phi: float
    shaw_chi: float
    shaw_noise: float
    uav_max_stops_per_trip: int = 4
    traffic_provider: any = None
    current_time_minutes: float = 0.0
    service_time_min: float = 0.0
    service_time_max: float = 0.0
    
    def get_dynamic_service_time(self, node, arrival_time):
        if self.traffic_provider is None:
            return self.service_time_min
        risk = self.traffic_provider.get_node_risk(arrival_time, node)
        return self.service_time_min + (risk / 10.0) * (self.service_time_max - self.service_time_min)


def build_runtime_context(
    dist_uav,
    hub_indices,
    target_indices,
    w_risk,
    w_cover,
    t_max,
    uav_endurance,
    max_uav_trips,
    num_cars,
    num_uavs,
    shaw_phi,
    shaw_chi,
    shaw_noise,
    uav_max_stops_per_trip=4,
    traffic_provider=None,
    current_time_minutes=0.0,
    service_time_min=0.0,
    service_time_max=0.0,
):
    return RuntimeContext(
        dist_uav=dist_uav,
        hub_indices=list(hub_indices),
        target_indices=list(target_indices),
        w_risk=w_risk,
        w_cover=w_cover,
        t_max=t_max,
        uav_endurance=uav_endurance,
        max_uav_trips=max_uav_trips,
        num_cars=num_cars,
        num_uavs=num_uavs,
        shaw_phi=shaw_phi,
        shaw_chi=shaw_chi,
        shaw_noise=shaw_noise,
        uav_max_stops_per_trip=uav_max_stops_per_trip,
        traffic_provider=traffic_provider,
        current_time_minutes=current_time_minutes,
        service_time_min=service_time_min,
        service_time_max=service_time_max,
    )


class PatrolState(State):
    def __init__(self, car_routes, uav_trips, unassigned, context, car_durations=None):
        self.car_routes = car_routes
        self.uav_trips = uav_trips
        self.unassigned = unassigned
        self.context = context
        self.node_arrival_times = {}

        if car_durations is None:
            self.car_durations = self._recalc_durations()
        else:
            self.car_durations = car_durations

        self._objective = None

    def _recalc_durations(self):
        durs = []
        self.node_arrival_times.clear()
        for r in self.car_routes:
            current_t = self.context.current_time_minutes
            for i in range(len(r) - 1):
                if self.context.traffic_provider is not None:
                    travel_time = self.context.traffic_provider.get_car_travel_time(current_t, r[i], r[i + 1])
                else:
                    travel_time = 0.0
                current_t += travel_time
                next_node = r[i + 1]
                self.node_arrival_times[next_node] = current_t
                if next_node not in self.context.hub_indices:
                    service_time = self.context.get_dynamic_service_time(next_node, current_t)
                    current_t += service_time
            total_duration = current_t - self.context.current_time_minutes
            durs.append(total_duration)
        return durs

    def copy(self):
        new_state = PatrolState(
            [list(r) for r in self.car_routes],
            [list(t) for t in self.uav_trips],
            list(self.unassigned),
            self.context,
            list(self.car_durations),
        )
        new_state.node_arrival_times = self.node_arrival_times.copy()
        return new_state

    def objective(self):
        if self._objective is None:
            r_sum, cover_count = self.calc_stats()
            self._objective = -(r_sum * self.context.w_risk + cover_count * self.context.w_cover)
        return self._objective

    def calc_stats(self):
        visited = set()
        risk_sum = 0.0
        
        # 计算警车路径的风险（动态查询）
        for r in self.car_routes:
            current_t = self.context.current_time_minutes
            for i in range(len(r) - 1):
                # 从当前节点到下一个节点
                current_node = r[i]
                next_node = r[i + 1]
                
                # 计算行驶时间并更新当前时间
                if self.context.traffic_provider is not None:
                    travel_time = self.context.traffic_provider.get_car_travel_time(current_t, current_node, next_node)
                else:
                    travel_time = 0.0
                current_t += travel_time
                
                # 如果下一个节点不是 hub，计算风险并加上服务时间
                if next_node not in self.context.hub_indices:
                    if next_node not in visited:
                        visited.add(next_node)
                        # 在到达节点的那一刻查询风险
                        if self.context.traffic_provider is not None:
                            risk = self.context.traffic_provider.get_node_risk(current_t, next_node)
                        else:
                            risk = 0.0
                        risk_sum += risk
                    # 加上动态服务时间
                    service_time = self.context.get_dynamic_service_time(next_node, current_t)
                    current_t += service_time
        
        # 计算无人机路径的风险（使用基准时间）
        for t in self.uav_trips:
            for n in t:
                if n not in self.context.hub_indices and n not in visited:
                    visited.add(n)
                    if self.context.traffic_provider is not None:
                        risk = self.context.traffic_provider.get_node_risk(self.context.current_time_minutes, n)
                    else:
                        risk = 0.0
                    risk_sum += risk
        
        return risk_sum, len(visited)


def uav_trip_time(trip, context, current_time=None, traffic_matrix=None):
    if not trip or len(trip) < 2:
        return 0.0
    _ = current_time
    _ = traffic_matrix
    fly = sum(context.dist_uav[trip[i], trip[i + 1]] for i in range(len(trip) - 1))
    svc = sum(context.get_dynamic_service_time(trip[i], context.current_time_minutes) for i in range(1, len(trip) - 1))
    return float(fly + svc)


def check_insert_uav_trip(trip, node, pos, context, current_time=None, traffic_matrix=None):
    if pos <= 0 or pos >= len(trip):
        return False, 0.0
    prev_n = trip[pos - 1]
    next_n = trip[pos]
    service_time = context.get_dynamic_service_time(node, context.current_time_minutes)
    delta = (
        context.dist_uav[prev_n, node]
        + context.dist_uav[node, next_n]
        - context.dist_uav[prev_n, next_n]
        + service_time
    )
    if uav_trip_time(trip, context, current_time=current_time, traffic_matrix=traffic_matrix) + delta > context.uav_endurance:
        return False, float(delta)
    return True, float(delta)


def check_insert_turbo(state, car_idx, node, pos, context, current_time=None, traffic_matrix=None):
    _ = current_time
    _ = traffic_matrix
    route = state.car_routes[car_idx]
    current_duration = state.car_durations[car_idx]
    
    # 构建新路径
    new_route = route[:pos] + [node] + route[pos:]
    
    # 对新路径进行完整的时间推演
    current_t = context.current_time_minutes
    for i in range(len(new_route) - 1):
        if context.traffic_provider is not None:
            travel_time = context.traffic_provider.get_car_travel_time(current_t, new_route[i], new_route[i+1])
        else:
            travel_time = 0.0
        current_t += travel_time
        next_node = new_route[i+1]
        if next_node not in context.hub_indices:
            service_time = context.get_dynamic_service_time(next_node, current_t)
            current_t += service_time
    new_duration = current_t - context.current_time_minutes
    
    delta = new_duration - current_duration
    if new_duration > context.t_max:
        return False, delta
    return True, delta


def destroy_random(state, rnd, context, current_time=None, traffic_matrix=None):
    _ = current_time
    _ = traffic_matrix
    s = state.copy()
    assigned = []
    for ri, r in enumerate(s.car_routes):
        for i, n in enumerate(r):
            if n not in context.hub_indices:
                assigned.append(('c', ri, i, n))
    for ti, t in enumerate(s.uav_trips):
        for i, n in enumerate(t):
            if n not in context.hub_indices:
                assigned.append(('u', ti, i, n))

    if len(assigned) < 2:
        return s
    lower_bound = min(20, max(1, len(assigned) // 2))
    upper_bound = min(len(assigned), 60)
    # 安全生成 k
    if lower_bound < upper_bound:
        k = rnd.randint(lower_bound, upper_bound)
    else:
        k = lower_bound
    indices = rnd.choice(len(assigned), k, replace=False)
    to_remove = [assigned[i] for i in indices]
    rem_ids = set([x[3] for x in to_remove])

    s.car_routes = [[n for n in r if n not in rem_ids] for r in s.car_routes]
    s.uav_trips = [[n for n in t if n not in rem_ids] for t in s.uav_trips if len(t) > 2]
    s.unassigned.extend(list(rem_ids))
    s.car_durations = s._recalc_durations()
    s._objective = None
    return s


def destroy_worst(state, rnd, context, current_time=None, traffic_matrix=None):
    _ = current_time
    _ = traffic_matrix
    s = state.copy()
    assigned = []
    for r in s.car_routes:
        for n in r:
            if n not in context.hub_indices:
                assigned.append(n)
    if len(assigned) < 2:
        return s
    if context.traffic_provider is not None:
        assigned.sort(key=lambda x: context.traffic_provider.get_node_risk(context.current_time_minutes, x))
    lower_bound = min(10, max(1, len(assigned) // 2))
    upper_bound = min(len(assigned), 50)
    # 安全生成 k
    if lower_bound < upper_bound:
        k = rnd.randint(lower_bound, upper_bound)
    else:
        k = lower_bound
    rem_ids = set(assigned[:k])

    s.car_routes = [[n for n in r if n not in rem_ids] for r in s.car_routes]
    s.uav_trips = [[n for n in t if n not in rem_ids] for t in s.uav_trips if len(t) > 2]
    s.unassigned.extend(list(rem_ids))
    s.car_durations = s._recalc_durations()
    s._objective = None
    return s


def destroy_shaw(state, rnd, context, current_time=None, traffic_matrix=None):
    _ = current_time
    _ = traffic_matrix
    s = state.copy()
    assigned = []
    for r in s.car_routes:
        for n in r:
            if n not in context.hub_indices:
                assigned.append(n)

    if len(assigned) < 2:
        return s

    pivot = rnd.choice(assigned)
    assigned.remove(pivot)
    removed = [pivot]
    # 动态适应规模
    lower_bound = min(5, max(1, len(assigned) // 3))
    upper_bound = min(len(assigned) + 1, 15)
    if lower_bound < upper_bound:
        k = rnd.randint(lower_bound, upper_bound)
    else:
        k = lower_bound

    while len(removed) < k and assigned:
        pivot = rnd.choice(removed)
        candidates = []
        t_pivot = state.node_arrival_times.get(pivot, context.current_time_minutes)
        for n in assigned:
            t_n = state.node_arrival_times.get(n, context.current_time_minutes)
            # 使用两个节点的到达时间的平均值作为查询时间
            query_time = (t_pivot + t_n) / 2
            if context.traffic_provider is not None:
                dist = context.traffic_provider.get_car_travel_time(query_time, pivot, n)
                risk_pivot = context.traffic_provider.get_node_risk(t_pivot, pivot)
                risk_n = context.traffic_provider.get_node_risk(t_n, n)
                risk_diff = abs(risk_pivot - risk_n)
            else:
                dist = 0.0
                risk_diff = 0.0
            R = context.shaw_phi * dist + context.shaw_chi * risk_diff
            candidates.append((n, R))

        candidates.sort(key=lambda x: x[1])
        idx = int(pow(rnd.random_sample(), context.shaw_noise) * len(candidates))
        chosen = candidates[min(idx, len(candidates) - 1)][0]
        removed.append(chosen)
        assigned.remove(chosen)

    rem_ids = set(removed)
    s.car_routes = [[n for n in r if n not in rem_ids] for r in s.car_routes]
    s.uav_trips = [[n for n in t if n not in rem_ids] for t in s.uav_trips if len(t) > 2]
    s.unassigned.extend(list(rem_ids))
    s.car_durations = s._recalc_durations()
    s._objective = None
    return s


def repair_greedy(state, rnd, context, current_time=None, traffic_matrix=None):
    s = state.copy()
    candidates = s.unassigned[:]
    rnd.shuffle(candidates)
    batch = candidates[:50]

    for node in batch:
        best_gain = float('-inf')
        best_act = None
        if context.traffic_provider is not None:
            node_risk = context.traffic_provider.get_node_risk(context.current_time_minutes, node)
        else:
            node_risk = 0.0

        for ri in range(context.num_cars):
            route = s.car_routes[ri]
            # 第一阶段：O(1) 粗筛
            static_deltas = []
            for i in range(1, len(route)):
                prev_n = route[i - 1]
                next_n = route[i]
                # 使用基准时间查询静态截面距离
                if context.traffic_provider is not None:
                    dist_prev_node = context.traffic_provider.get_car_travel_time(context.current_time_minutes, prev_n, node)
                    dist_node_next = context.traffic_provider.get_car_travel_time(context.current_time_minutes, node, next_n)
                    dist_prev_next = context.traffic_provider.get_car_travel_time(context.current_time_minutes, prev_n, next_n)
                else:
                    dist_prev_node = 0.0
                    dist_node_next = 0.0
                    dist_prev_next = 0.0
                static_delta = dist_prev_node + dist_node_next - dist_prev_next
                static_deltas.append((i, static_delta))
            
            # 只保留 static_delta 最小的前 3 个位置
            static_deltas.sort(key=lambda x: x[1])
            top_positions = static_deltas[:3]
            
            # 第二阶段：O(N) 精算
            for pos, _ in top_positions:
                valid, delta = check_insert_turbo(
                    s,
                    ri,
                    node,
                    pos,
                    context,
                    current_time=current_time,
                    traffic_matrix=traffic_matrix,
                )
                if valid:
                    gain = node_risk * context.w_risk - delta * 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_act = ('c', ri, pos, delta)

        for ti, trip in enumerate(s.uav_trips):
            num_stops = len(trip) - 2
            if num_stops >= context.uav_max_stops_per_trip:
                continue
            for pos in range(1, len(trip)):
                valid_u, delta_u = check_insert_uav_trip(
                    trip,
                    node,
                    pos,
                    context,
                    current_time=current_time,
                    traffic_matrix=traffic_matrix,
                )
                if valid_u:
                    gain = node_risk * context.w_risk - delta_u * 0.05 + 3.0
                    if gain > best_gain:
                        best_gain = gain
                        best_act = ('u_ins', ti, pos, delta_u)

        if len(s.uav_trips) < context.max_uav_trips:
            dists = [context.dist_uav[h, node] for h in context.hub_indices]
            h_idx = context.hub_indices[int(np.argmin(dists))]
            service_time = context.get_dynamic_service_time(node, context.current_time_minutes)
            ft = context.dist_uav[h_idx, node] * 2 + service_time
            if ft <= context.uav_endurance:
                gain = node_risk * context.w_risk + 5
                if gain > best_gain:
                    best_gain = gain
                    best_act = ('u_new', h_idx)

        if best_act:
            if best_act[0] == 'c':
                _, ri, i, delta = best_act
                s.car_routes[ri].insert(i, node)
                s.car_durations[ri] += delta
            elif best_act[0] == 'u_ins':
                _, ti, pos, _ = best_act
                s.uav_trips[ti].insert(pos, node)
            else:
                _, h = best_act
                s.uav_trips.append([h, node, h])
            s.unassigned.remove(node)

    s._objective = None
    return s


def repair_regret(state, rnd, context, current_time=None, traffic_matrix=None):
    s = state.copy()
    candidates = s.unassigned[:]
    rnd.shuffle(candidates)
    batch = candidates[:50]

    while batch:
        regrets = []
        for node in batch:
            best_sol = (float('inf'), None)
            second_sol = (float('inf'), None)
            if context.traffic_provider is not None:
                node_risk = context.traffic_provider.get_node_risk(context.current_time_minutes, node)
            else:
                node_risk = 0.0

            for ri in range(context.num_cars):
                route = s.car_routes[ri]
                # 第一阶段：O(1) 粗筛
                static_deltas = []
                for i in range(1, len(route)):
                    prev_n = route[i - 1]
                    next_n = route[i]
                    # 使用基准时间查询静态截面距离
                    if context.traffic_provider is not None:
                        dist_prev_node = context.traffic_provider.get_car_travel_time(context.current_time_minutes, prev_n, node)
                        dist_node_next = context.traffic_provider.get_car_travel_time(context.current_time_minutes, node, next_n)
                        dist_prev_next = context.traffic_provider.get_car_travel_time(context.current_time_minutes, prev_n, next_n)
                    else:
                        dist_prev_node = 0.0
                        dist_node_next = 0.0
                        dist_prev_next = 0.0
                    static_delta = dist_prev_node + dist_node_next - dist_prev_next
                    static_deltas.append((i, static_delta))
                
                # 只保留 static_delta 最小的前 3 个位置
                static_deltas.sort(key=lambda x: x[1])
                top_positions = static_deltas[:3]
                
                # 第二阶段：O(N) 精算
                for pos, _ in top_positions:
                    valid, delta = check_insert_turbo(
                        s,
                        ri,
                        node,
                        pos,
                        context,
                        current_time=current_time,
                        traffic_matrix=traffic_matrix,
                    )
                    if valid:
                        score = delta - node_risk * context.w_risk
                        act = ('c', ri, pos, delta)
                        if score < best_sol[0]:
                            second_sol = best_sol
                            best_sol = (score, act)
                        elif score < second_sol[0]:
                            second_sol = (score, act)

            if best_sol[1] is None:
                continue
            regret_val = second_sol[0] - best_sol[0]
            if second_sol[1] is None:
                regret_val = 9999
            regrets.append((regret_val, node, best_sol[1]))

        if not regrets:
            break

        regrets.sort(key=lambda x: x[0], reverse=True)
        _, node, act = regrets[0]
        _, ri, i, delta = act
        s.car_routes[ri].insert(i, node)
        s.car_durations[ri] += delta

        if node in s.unassigned:
            s.unassigned.remove(node)
            batch.remove(node)

    s._objective = None
    return s
