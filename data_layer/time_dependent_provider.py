import numpy as np
from typing import Union, Optional


class TimeDependentTrafficProvider:
    _warned_inf_pairs = set()
    """
    时间依赖的交通数据提供器
    
    该类负责加载和提供时空张量数据，供ALNS算法进行动态巡逻路径规划
    支持根据当前时间查询时变的交通数据和风险值
    
    时空张量 Shape 说明：
    - risk_tensor: (288, num_nodes) - 288个时间槽的节点风险值
    - car_time_tensor: (288, num_nodes, num_nodes) - 288个时间槽的车辆行驶时间矩阵
    - dist_matrix_uav: (num_nodes, num_nodes) - 无人机飞行时间矩阵（静态）
    - coords_array: (num_nodes, 2) - 节点坐标数组
    - node_map: dict - 原始节点ID到连续索引的映射
    """
    
    # 单例实例
    _instance = None
    
    def __new__(cls, npz_path: Optional[str] = None):
        """
        单例模式实现
        确保类在内存中只被实例化一次
        """
        if cls._instance is None and npz_path is not None:
            cls._instance = super(TimeDependentTrafficProvider, cls).__new__(cls)
            cls._instance._initialize(npz_path)
        return cls._instance
    
    def _initialize(self, npz_path: str):
        """
        初始化方法，加载时空张量数据
        
        Args:
            npz_path: 时空张量文件路径
        """
        print(f"加载时空张量数据: {npz_path}")
        
        # 加载数据
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            raise Exception(f"加载时空张量失败: {e}")
        
        # 存储数据为实例属性
        self.node_map = data['node_map'].item()  # 转换为Python字典
        self.coords_array = data['coords_array']
        self.dist_matrix_uav = data['dist_matrix_uav']
        self.risk_tensor = data['risk_tensor']
        self.car_time_tensor = data['car_time_tensor']
        
        # 计算节点数量
        self.num_nodes = len(self.node_map)
        
        # 定义时间步长（5分钟一个时间槽）
        self.TIME_SLOT_MINUTES = 5
        
        # 时间槽总数
        self.TOTAL_TIME_SLOTS = 288
        
        print(f"时空张量加载完成: {self.num_nodes} 个节点, {self.TOTAL_TIME_SLOTS} 个时间槽")
    
    def _get_time_index(self, current_time_minutes: float) -> int:
        """
        根据当前时间计算对应的时间槽索引
        
        Args:
            current_time_minutes: 从00:00开始的相对分钟数
            
        Returns:
            int: 时间槽索引 (0-287)
        """
        if current_time_minutes is None or not np.isfinite(current_time_minutes):
            raise ValueError(f"当前时间无效: {current_time_minutes}")

        # 计算时间槽索引
        idx = int(float(current_time_minutes) // self.TIME_SLOT_MINUTES)
        
        # 处理越界情况，确保索引在有效范围内
        idx = min(idx, self.TOTAL_TIME_SLOTS - 1)
        idx = max(idx, 0)
        
        return idx
    
    def _get_node_index(self, node: Union[str, int]) -> int:
        """
        获取节点的连续索引
        
        Args:
            node: 节点ID（字符串）或连续索引（整数）
            
        Returns:
            int: 节点的连续索引
        """
        if isinstance(node, int):
            # 如果已经是连续索引，直接返回
            if 0 <= node < self.num_nodes:
                return node
            else:
                raise ValueError(f"节点索引 {node} 超出范围")
        elif isinstance(node, str):
            # 如果是原始节点ID，通过node_map映射
            if node in self.node_map:
                return self.node_map[node]
            else:
                raise ValueError(f"节点ID {node} 不存在")
        else:
            raise TypeError(f"节点类型错误，期望 str 或 int，得到 {type(node)}")
    
    def get_car_travel_time(self, current_time_minutes: float, node_i: Union[str, int], node_j: Union[str, int]) -> float:
        """
        获取车辆在指定时间从节点i到节点j的行驶时间
        
        Args:
            current_time_minutes: 当前时间（从00:00开始的分钟数）
            node_i: 起点节点（ID或索引）
            node_j: 终点节点（ID或索引）
            
        Returns:
            float: 行驶时间（分钟）
        """
        try:
            # 获取时间槽索引
            time_idx = self._get_time_index(current_time_minutes)
            
            # 获取节点索引
            i = self._get_node_index(node_i)
            j = self._get_node_index(node_j)
            
            # 返回行驶时间
            travel_time = float(self.car_time_tensor[time_idx, i, j])
            if not np.isfinite(travel_time):
                key = (time_idx, i, j)
                if key not in self._warned_inf_pairs:
                    print(f"获取车辆行驶时间失败: 车辆行驶时间无效: {travel_time}")
                    self._warned_inf_pairs.add(key)
                return 999999.0
            return travel_time
        except Exception as e:
            print(f"获取车辆行驶时间失败: {e}")
            # 返回一个较大的惩罚距离
            return 999999.0
    
    def get_uav_travel_time(self, node_i: Union[str, int], node_j: Union[str, int]) -> float:
        """
        获取无人机从节点i到节点j的飞行时间
        无人机不受交通拥堵影响，使用静态距离矩阵
        
        Args:
            node_i: 起点节点（ID或索引）
            node_j: 终点节点（ID或索引）
            
        Returns:
            float: 飞行时间（分钟）
        """
        try:
            # 获取节点索引
            i = self._get_node_index(node_i)
            j = self._get_node_index(node_j)
            
            # 返回飞行时间
            return self.dist_matrix_uav[i, j]
        except Exception as e:
            print(f"获取无人机飞行时间失败: {e}")
            # 返回一个较大的惩罚距离
            return 999999.0
    
    def get_node_risk(self, current_time_minutes: float, node: Union[str, int]) -> float:
        """
        获取指定时间节点的风险值
        
        Args:
            current_time_minutes: 当前时间（从00:00开始的分钟数）
            node: 节点（ID或索引）
            
        Returns:
            float: 节点风险值（0-10）
        """
        try:
            # 获取时间槽索引
            time_idx = self._get_time_index(current_time_minutes)
            
            # 获取节点索引
            i = self._get_node_index(node)
            
            # 返回风险值
            risk = float(self.risk_tensor[time_idx, i])
            if not np.isfinite(risk):
                raise ValueError(f"节点风险值无效: {risk}")
            return risk
        except Exception as e:
            print(f"获取节点风险值失败: {e}")
            # 返回默认风险值
            return 5.0
    
    def get_node_coords(self, node: Union[str, int]) -> tuple:
        """
        获取节点的坐标
        
        Args:
            node: 节点（ID或索引）
            
        Returns:
            tuple: (x, y) 坐标
        """
        try:
            # 获取节点索引
            i = self._get_node_index(node)
            
            # 返回坐标
            return tuple(self.coords_array[i])
        except Exception as e:
            print(f"获取节点坐标失败: {e}")
            # 返回默认坐标
            return (0.0, 0.0)
    
    def get_all_nodes(self) -> list:
        """
        获取所有节点ID列表
        
        Returns:
            list: 节点ID列表
        """
        return list(self.node_map.keys())
    
    def get_num_nodes(self) -> int:
        """
        获取节点数量
        
        Returns:
            int: 节点数量
        """
        return self.num_nodes


# 工厂函数，用于获取单例实例
def get_traffic_provider(npz_path: Optional[str] = None) -> TimeDependentTrafficProvider:
    """
    获取交通数据提供器的单例实例
    
    Args:
        npz_path: 时空张量文件路径（首次调用时需要）
        
    Returns:
        TimeDependentTrafficProvider: 交通数据提供器实例
    """
    return TimeDependentTrafficProvider(npz_path)
