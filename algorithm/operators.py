from dataclasses import dataclass

import numpy as np

try:
    from alns import State
except ImportError:
    class State:
        pass


@dataclass
class RuntimeContext:
    dist_car: np.ndarray
    dist_uav: np.ndarray
    nodes_risk: np.ndarray
    hub_indices: list
    target_indices: list
    node_service_time: np.ndarray
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
    current_time: object = None
    traffic_matrix: object = None


def build_runtime_context(
    dist_car,
    dist_uav,
    nodes_risk,
    hub_indices,
    target_indices,
    node_service_time,
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
    current_time=None,
    traffic_matrix=None,
):
    return RuntimeContext(
        dist_car=dist_car,
        dist_uav=dist_uav,
        nodes_risk=nodes_risk,
        hub_indices=list(hub_indices),
        target_indices=list(target_indices),
        node_service_time=node_service_time,
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
        current_time=current_time,
        traffic_matrix=traffic_matrix,
    )


class PatrolState(State):
    def __init__(self, car_routes, uav_trips, unassigned, context, car_durations=None):
        self.car_routes = car_routes
        self.uav_trips = uav_trips
        self.unassigned = unassigned
        self.context = context

        if car_durations is None:
            self.car_durations = self._recalc_durations()
        else:
            self.car_durations = car_durations

        self._objective = None

    def _recalc_durations(self):
        durs = []
        for r in self.car_routes:
            t = 0
            for i in range(len(r) - 1):
                t += self.context.dist_car[r[i], r[i + 1]]
                if r[i + 1] not in self.context.hub_indices:
                    t += self.context.node_service_time[r[i + 1]]
            durs.append(t)
        return durs

    def copy(self):
        return PatrolState(
            [list(r) for r in self.car_routes],
            [list(t) for t in self.uav_trips],
            list(self.unassigned),
            self.context,
            list(self.car_durations),
        )

    def objective(self):
        if self._objective is None:
            r_sum, cover_count = self.calc_stats()
            self._objective = -(r_sum * self.context.w_risk + cover_count * self.context.w_cover)
        return self._objective

    def calc_stats(self):
        visited = set()
        for r in self.car_routes:
            for n in r:
                if n not in self.context.hub_indices:
                    visited.add(n)
        for t in self.uav_trips:
            for n in t:
                if n not in self.context.hub_indices:
                    visited.add(n)
        return sum(self.context.nodes_risk[list(visited)]), len(visited)


def uav_trip_time(trip, context, current_time=None, traffic_matrix=None):
    if not trip or len(trip) < 2:
        return 0.0
    _ = current_time
    _ = traffic_matrix
    fly = sum(context.dist_uav[trip[i], trip[i + 1]] for i in range(len(trip) - 1))
    svc = sum(context.node_service_time[trip[i]] for i in range(1, len(trip) - 1))
    return float(fly + svc)


def check_insert_uav_trip(trip, node, pos, context, current_time=None, traffic_matrix=None):
    if pos <= 0 or pos >= len(trip):
        return False, 0.0
    prev_n = trip[pos - 1]
    next_n = trip[pos]
    delta = (
        context.dist_uav[prev_n, node]
        + context.dist_uav[node, next_n]
        - context.dist_uav[prev_n, next_n]
        + context.node_service_time[node]
    )
    if uav_trip_time(trip, context, current_time=current_time, traffic_matrix=traffic_matrix) + delta > context.uav_endurance:
        return False, float(delta)
    return True, float(delta)


def check_insert_turbo(state, car_idx, node, pos, context, current_time=None, traffic_matrix=None):
    _ = current_time
    _ = traffic_matrix
    route = state.car_routes[car_idx]
    current_duration = state.car_durations[car_idx]
    prev_n = route[pos - 1]
    next_n = route[pos]
    delta = (
        context.dist_car[prev_n, node]
        + context.dist_car[node, next_n]
        - context.dist_car[prev_n, next_n]
        + context.node_service_time[node]
    )
    if current_duration + delta > context.t_max:
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

    if len(assigned) < 5:
        return s

    k = rnd.randint(20, min(len(assigned), 60))
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
    if not assigned:
        return s

    assigned.sort(key=lambda x: context.nodes_risk[x])
    k = rnd.randint(10, min(len(assigned), 50))
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

    if len(assigned) < 5:
        return s

    pivot = rnd.choice(assigned)
    assigned.remove(pivot)
    removed = [pivot]
    k = rnd.randint(5, min(len(assigned) + 1, 15))

    while len(removed) < k and assigned:
        pivot = rnd.choice(removed)
        candidates = []
        for n in assigned:
            dist = context.dist_car[pivot, n]
            risk_diff = abs(context.nodes_risk[pivot] - context.nodes_risk[n])
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

        for ri in range(context.num_cars):
            route = s.car_routes[ri]
            step = 2 if len(route) > 8 else 1
            for i in range(1, len(route), step):
                valid, delta = check_insert_turbo(
                    s,
                    ri,
                    node,
                    i,
                    context,
                    current_time=current_time,
                    traffic_matrix=traffic_matrix,
                )
                if valid:
                    gain = context.nodes_risk[node] * context.w_risk - delta * 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_act = ('c', ri, i, delta)

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
                    gain = context.nodes_risk[node] * context.w_risk - delta_u * 0.05 + 3.0
                    if gain > best_gain:
                        best_gain = gain
                        best_act = ('u_ins', ti, pos, delta_u)

        if len(s.uav_trips) < context.max_uav_trips:
            dists = [context.dist_uav[h, node] for h in context.hub_indices]
            h_idx = context.hub_indices[int(np.argmin(dists))]
            ft = context.dist_uav[h_idx, node] * 2 + context.node_service_time[node]
            if ft <= context.uav_endurance:
                gain = context.nodes_risk[node] * context.w_risk + 5
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

            for ri in range(context.num_cars):
                route = s.car_routes[ri]
                step = 2 if len(route) > 8 else 1
                for i in range(1, len(route), step):
                    valid, delta = check_insert_turbo(
                        s,
                        ri,
                        node,
                        i,
                        context,
                        current_time=current_time,
                        traffic_matrix=traffic_matrix,
                    )
                    if valid:
                        score = delta - context.nodes_risk[node] * context.w_risk
                        act = ('c', ri, i, delta)
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
