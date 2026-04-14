import pandas as pd
import numpy as np
import pickle
import psutil
import os
import tempfile
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

# 常量定义
T = 288  # 时间槽数量（24小时 × 60分钟 / 5分钟）

# 内存使用监控
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# 步骤1：读取数据
def load_data():
    """加载所有需要的数据"""
    # 读取路网数据
    net_df = pd.read_csv('data/all_net.csv')
    
    # 读取交通数据
    traffic_df = pd.read_csv('data/all_traffic_data.csv')
    
    # 读取缓存数据
    with open('data/map_data_from_csv.pkl', 'rb') as f:
        cached_data = pickle.load(f)
    
    return net_df, traffic_df, cached_data

# 步骤2：处理交通时间序列
def process_traffic_data(traffic_df, net_df):
    """处理交通数据，生成每路段的时序速度曲线"""
    # 按 link_id 分组，处理速度数据
    link_speed_dict = {}
    
    # 先构建路段到base_speed的映射，加速查询
    base_speed_map = {row['link_id']: row['base_speed'] for _, row in net_df.iterrows()}
    
    # 获取所有路段数量
    total_links = len(traffic_df['link_id'].unique())
    processed_links = 0
    
    print(f"开始处理 {total_links} 条路段的交通数据...")
    
    for link_id, group in traffic_df.groupby('link_id'):
        # 收集所有天的速度数据
        all_speed_series = []
        avg_speed_value = None
        
        for _, row in group.iterrows():
            try:
                # 分割速度字符串并转换为浮点数
                speeds = list(map(float, row['speed'].split(';')))
                if len(speeds) == T:
                    all_speed_series.append(speeds)
                # 记录avg_speed值
                if avg_speed_value is None:
                    avg_speed_value = row['avg_speed']
            except (ValueError, AttributeError):
                # 处理解析错误
                pass
        
        # 计算平均速度曲线
        if all_speed_series:
            # 对所有天的速度求平均
            avg_speed_curve = np.mean(all_speed_series, axis=0)
        else:
            # 如果没有有效数据，使用 avg_speed 或 base_speed
            if link_id in base_speed_map:
                avg_speed = base_speed_map[link_id]
            else:
                # 如果路网中也没有信息，使用交通数据中的 avg_speed
                if avg_speed_value is not None:
                    avg_speed = avg_speed_value
                else:
                    avg_speed = 40.0  # 默认值
            avg_speed_curve = np.full(T, avg_speed)
        
        link_speed_dict[link_id] = avg_speed_curve
        processed_links += 1
        
        # 每处理50条路段打印一次进度
        if processed_links % 50 == 0:
            progress = (processed_links / total_links) * 100
            print(f"处理了 {processed_links}/{total_links} 条路段 ({progress:.1f}%)")
    
    print(f"交通数据处理完成，共处理了 {len(link_speed_dict)} 条路段")
    return link_speed_dict

# 步骤3：构建时变节点风险矩阵
def build_risk_tensor(link_speed_dict, net_df, node_map, num_nodes):
    """构建时变节点风险矩阵"""
    risk_tensor = np.zeros((T, num_nodes))
    
    # 首先计算每个路段在每个时间片的风险
    link_risk_dict = {}
    for link_id, speeds in link_speed_dict.items():
        # 获取路段信息
        link_info = net_df[net_df['link_id'] == link_id]
        if link_info.empty:
            continue
        
        base_speed = link_info.iloc[0]['base_speed']
        start_node = str(link_info.iloc[0]['start_node_id'])
        end_node = str(link_info.iloc[0]['end_node_id'])
        
        # 检查节点是否在映射中
        if start_node not in node_map or end_node not in node_map:
            continue
        
        start_idx = node_map[start_node]
        end_idx = node_map[end_node]
        
        # 计算每个时间片的风险
        link_risks = []
        for v_t in speeds:
            # 计算损失率
            loss_rate = max(0, (base_speed - v_t) / base_speed)
            # 计算流量代理
            flow_proxy = 1 / max(5, v_t)
            # 计算路段风险
            risk_link = loss_rate * flow_proxy
            link_risks.append(risk_link)
        
        link_risk_dict[link_id] = (start_idx, end_idx, link_risks)
    
    # 将路段风险映射到节点
    for link_id, (start_idx, end_idx, link_risks) in link_risk_dict.items():
        for t in range(T):
            risk = link_risks[t]
            # 平均分配到两个节点
            risk_tensor[t, start_idx] += risk
            risk_tensor[t, end_idx] += risk
    
    # 对每个时间片进行Min-Max归一化到[0, 10]
    for t in range(T):
        row = risk_tensor[t]
        min_val = np.min(row)
        max_val = np.max(row)
        if max_val > min_val:
            risk_tensor[t] = 10 * (row - min_val) / (max_val - min_val)
        else:
            risk_tensor[t] = 0  # 所有值相同，设为0
    
    return risk_tensor

# 步骤4：构建时变地面警车耗时张量
def build_car_time_tensor(link_speed_dict, net_df, node_map, num_nodes):
    """构建时变地面警车耗时张量"""
    # 构建路段到节点索引的映射
    link_to_nodes = {}
    valid_links = 0
    for _, row in net_df.iterrows():
        link_id = row['link_id']
        start_node = str(row['start_node_id'])
        end_node = str(row['end_node_id'])
        road_length = row['road_length']
        
        if start_node in node_map and end_node in node_map:
            start_idx = node_map[start_node]
            end_idx = node_map[end_node]
            link_to_nodes[link_id] = (start_idx, end_idx, road_length)
            valid_links += 1
    
    print(f"有效路段数量: {valid_links}")
    
    # 使用内存映射文件来存储大型张量
    shape = (T, num_nodes, num_nodes)
    dtype = np.float16  # 使用float16减少内存使用
    
    # 创建临时文件用于内存映射
    temp_dir = tempfile.gettempdir()
    temp_filename = os.path.join(temp_dir, 'car_time_tensor_temp.dat')
    
    print(f"创建内存映射文件: {temp_filename}")
    
    # 计算文件大小
    itemsize = np.dtype(dtype).itemsize
    filesize = int(np.prod(shape)) * itemsize
    
    print(f"张量形状: {shape}, 数据类型: {dtype}, 文件大小: {filesize / (1024**3):.2f} GB")
    
    # 创建文件
    with open(temp_filename, 'wb') as f:
        f.seek(filesize - 1)
        f.write(b'\x00')
    
    # 创建内存映射数组
    car_time_tensor = np.memmap(temp_filename, dtype=dtype, mode='r+', shape=shape)
    
    # 分批次处理时间片，每批次处理12个时间片
    batch_size = 12
    
    try:
        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)
            print(f"处理批次 {batch_start//batch_size + 1}/{(T + batch_size - 1)//batch_size}, 时间片 {batch_start}-{batch_end-1}")
            
            for t in range(batch_start, batch_end):
                if t % 12 == 0:  # 每12个时间片打印一次进度
                    print(f"处理时间片 {t}/{T}, 内存使用: {get_memory_usage():.2f} MB")
                
                try:
                    # 构建当前时间片的邻接矩阵
                    row_indices = []
                    col_indices = []
                    data = []
                    
                    for link_id, (start_idx, end_idx, road_length) in link_to_nodes.items():
                        if link_id in link_speed_dict:
                            v_t = link_speed_dict[link_id][t]
                            # 计算通行时间（单位：秒）
                            # 假设 road_length 单位是米，v_t 单位是 km/h
                            # 转换为：时间（秒）= 距离（米） / 速度（米/秒）
                            speed_mps = v_t * 1000 / 3600
                            travel_time = road_length / max(1.0, speed_mps)
                            row_indices.append(start_idx)
                            col_indices.append(end_idx)
                            data.append(travel_time)
                    
                    # 创建稀疏矩阵
                    if data:
                        adj_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
                        # 计算所有节点对的最短路径
                        dist_matrix = shortest_path(csgraph=adj_matrix, directed=True, method='D')
                        car_time_tensor[t] = dist_matrix
                    else:
                        # 如果没有数据，使用无穷大
                        car_time_tensor[t] = np.inf
                except Exception as e:
                    print(f"处理时间片 {t} 时出错: {e}")
                    # 使用无穷大填充
                    car_time_tensor[t] = np.inf
            
            # 每批次结束后清理内存
            import gc
            gc.collect()
            print(f"批次完成，内存使用: {get_memory_usage():.2f} MB")
        
        # 将内存映射数组转换为常规数组
        print("将内存映射数组转换为常规数组...")
        car_time_tensor_result = np.array(car_time_tensor, dtype=np.float16)
        
        # 关闭内存映射
        del car_time_tensor
        import gc
        gc.collect()
        
    finally:
        # 删除临时文件
        if os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
                print(f"已删除临时文件: {temp_filename}")
            except PermissionError:
                print(f"无法删除临时文件: {temp_filename}，可能正在被其他进程使用")
    
    return car_time_tensor_result

# 主函数
def main():
    print("开始构建时空张量...")
    print(f"初始内存使用: {get_memory_usage():.2f} MB")
    
    # 加载数据
    net_df, traffic_df, cached_data = load_data()
    node_map = cached_data['node_map']
    coords_array = cached_data['coords_array']
    dist_matrix_uav = cached_data['dist_matrix_uav']
    num_nodes = len(node_map)
    
    print(f"节点数量: {num_nodes}")
    print(f"时间槽数量: {T}")
    print(f"加载数据后内存使用: {get_memory_usage():.2f} MB")
    
    # 处理交通数据
    print("处理交通数据...")
    link_speed_dict = process_traffic_data(traffic_df, net_df)
    print(f"处理了 {len(link_speed_dict)} 条路段的速度数据")
    print(f"处理交通数据后内存使用: {get_memory_usage():.2f} MB")
    
    # 构建风险张量
    print("构建风险张量...")
    risk_tensor = build_risk_tensor(link_speed_dict, net_df, node_map, num_nodes)
    print(f"风险张量形状: {risk_tensor.shape}")
    print(f"构建风险张量后内存使用: {get_memory_usage():.2f} MB")
    
    # 构建地面警车耗时张量
    print("构建地面警车耗时张量...")
    car_time_tensor = build_car_time_tensor(link_speed_dict, net_df, node_map, num_nodes)
    print(f"地面警车耗时张量形状: {car_time_tensor.shape}")
    print(f"构建地面警车耗时张量后内存使用: {get_memory_usage():.2f} MB")
    
    # 存储结果
    print("存储结果...")
    np.savez_compressed('data/dynamic_map_tensor.npz', 
             node_map=node_map,
             coords_array=coords_array,
             dist_matrix_uav=dist_matrix_uav,
             risk_tensor=risk_tensor,
             car_time_tensor=car_time_tensor)
    
    print("时空张量构建完成！")
    print(f"最终内存使用: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    main()
