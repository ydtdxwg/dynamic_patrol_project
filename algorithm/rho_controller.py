from .operators import PatrolState


class RHOController:
    def __init__(self, alns_engine, traffic_provider):
        self.alns_engine = alns_engine
        self.traffic_provider = traffic_provider
        self.current_sim_time = 0.0
        self.executed_routes = []

    def truncate_state(self, current_state, event_time):
        new_hub_indices = []
        pending_nodes = set()

        # 处理警车路线
        for route in current_state.car_routes:
            # 找到最后一个在 event_time 之前到达的节点
            last_visited_node = route[0]  # 初始化为起点
            for node in route:
                if node in current_state.node_arrival_times:
                    arrival_time = current_state.node_arrival_times[node]
                    if arrival_time <= event_time:
                        last_visited_node = node
                    else:
                        # 该节点未访问，加入待分配池
                        if node not in current_state.context.hub_indices:
                            pending_nodes.add(node)
                else:
                    # 如果节点没有到达时间，视为未访问
                    if node not in current_state.context.hub_indices:
                        pending_nodes.add(node)
            # 将最后访问的节点作为新的起点
            new_hub_indices.append(last_visited_node)

        # 处理无人机路线
        for trip in current_state.uav_trips:
            for node in trip:
                if node not in current_state.context.hub_indices:
                    # 简单处理：将所有无人机节点视为未访问
                    pending_nodes.add(node)

        # 确保 new_hub_indices 不为空
        if not new_hub_indices:
            # 如果没有警车路线，使用原始的 hub_indices
            new_hub_indices = current_state.context.hub_indices[:current_state.context.num_cars]

        # 确保 pending_nodes 只包含目标节点
        pending_nodes = pending_nodes.intersection(current_state.context.target_indices)

        return new_hub_indices, pending_nodes

    def handle_event_and_reschedule(self, current_state, event_time, event_type=None):
        # 更新内部时钟
        self.current_sim_time = event_time

        # 提取新的起点和待分配节点
        new_hub_indices, pending_nodes = self.truncate_state(current_state, event_time)

        # 保存原始的 hub_selector
        original_hub_selector = self.alns_engine.hub_selector

        # 创建临时的 hub_selector 函数，返回新的起点
        def temp_hub_selector(topology_bundle, current_time=None):
            return new_hub_indices

        # 临时替换 hub_selector
        self.alns_engine.hub_selector = temp_hub_selector

        # 记录已执行的路线
        self.executed_routes.append((event_time, current_state))

        # 调用 ALNS 引擎求解
        # 将 pending_nodes 转换为列表并传递给 solve 方法
        target_indices = list(pending_nodes)
        result = self.alns_engine.solve(current_time=self.current_sim_time, max_iterations=1000, target_indices=target_indices)
        best_state = result['best_state']

        # 恢复原始的 hub_selector
        self.alns_engine.hub_selector = original_hub_selector

        # 打印调度日志
        print(f"在时间 {event_time} 触发重调度，截获未访问节点 {len(pending_nodes)} 个，重新规划完毕")

        return best_state