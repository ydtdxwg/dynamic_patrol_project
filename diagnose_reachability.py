from pathlib import Path

import numpy as np


def resolve_npz_path() -> Path:
    project_dir = Path(__file__).resolve().parent
    return project_dir.parent / 'data' / 'dynamic_map_tensor.npz'


def load_tensor(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    node_map = data['node_map'].item()
    inverse_node_map = {idx: node_id for node_id, idx in node_map.items()}
    return {
        'node_map': node_map,
        'inverse_node_map': inverse_node_map,
        'risk_tensor': data['risk_tensor'],
        'car_time_tensor': data['car_time_tensor'],
        'dist_matrix_uav': data['dist_matrix_uav'],
        'coords_array': data['coords_array'],
    }


def summarize_time_slice(car_time_slice: np.ndarray, hub_idx: int, inverse_node_map: dict, slot_idx: int):
    num_nodes = car_time_slice.shape[0]
    finite_mask = np.isfinite(car_time_slice)
    diag_mask = np.eye(num_nodes, dtype=bool)
    offdiag_mask = ~diag_mask

    finite_offdiag = finite_mask & offdiag_mask
    inf_offdiag = (~finite_mask) & offdiag_mask

    total_pairs = int(offdiag_mask.sum())
    finite_pairs = int(finite_offdiag.sum())
    inf_pairs = int(inf_offdiag.sum())

    reachable_from_hub_mask = finite_mask[hub_idx].copy()
    reachable_to_hub_mask = finite_mask[:, hub_idx].copy()
    reachable_from_hub_mask[hub_idx] = True
    reachable_to_hub_mask[hub_idx] = True
    round_trip_mask = reachable_from_hub_mask & reachable_to_hub_mask

    isolated_out_mask = finite_mask.sum(axis=1) <= 1
    isolated_in_mask = finite_mask.sum(axis=0) <= 1

    unreachable_from_hub = np.where(~reachable_from_hub_mask)[0]
    unreachable_to_hub = np.where(~reachable_to_hub_mask)[0]
    round_trip_unreachable = np.where(~round_trip_mask)[0]
    isolated_out = np.where(isolated_out_mask)[0]
    isolated_in = np.where(isolated_in_mask)[0]

    def sample_node_ids(indices, limit=15):
        return [inverse_node_map.get(int(idx), str(int(idx))) for idx in indices[:limit]]

    print(f"\n=== 时间片 {slot_idx}（{slot_idx * 5} min）===")
    print(f"有限点对: {finite_pairs}/{total_pairs} ({finite_pairs / total_pairs:.2%})")
    print(f"不可达点对: {inf_pairs}/{total_pairs} ({inf_pairs / total_pairs:.2%})")
    print(f"Hub={hub_idx} 可达出去节点数: {int(reachable_from_hub_mask.sum())}/{num_nodes}")
    print(f"Hub={hub_idx} 可达返回节点数: {int(reachable_to_hub_mask.sum())}/{num_nodes}")
    print(f"Hub={hub_idx} 双向可达节点数: {int(round_trip_mask.sum())}/{num_nodes}")
    print(f"出度近似隔离节点数: {len(isolated_out)}")
    print(f"入度近似隔离节点数: {len(isolated_in)}")

    print(f"Hub 无法到达节点样本: {sample_node_ids(unreachable_from_hub)}")
    print(f"无法回到 Hub 的节点样本: {sample_node_ids(unreachable_to_hub)}")
    print(f"与 Hub 非双向连通节点样本: {sample_node_ids(round_trip_unreachable)}")


def main():
    npz_path = resolve_npz_path()
    print(f"加载诊断数据: {npz_path}")

    if not npz_path.exists():
        print('未找到 dynamic_map_tensor.npz，请检查路径。')
        return

    bundle = load_tensor(npz_path)
    car_time_tensor = bundle['car_time_tensor']
    inverse_node_map = bundle['inverse_node_map']

    num_slots, num_nodes, _ = car_time_tensor.shape
    hub_idx = 0

    print(f"节点数: {num_nodes}")
    print(f"时间片数: {num_slots}")

    global_finite = np.isfinite(car_time_tensor)
    global_diag = np.eye(num_nodes, dtype=bool)[None, :, :]
    global_offdiag = ~global_diag
    global_inf_ratio = ((~global_finite) & global_offdiag).sum() / global_offdiag.sum()
    print(f"全时段不可达点对占比: {global_inf_ratio:.2%}")

    key_slots = sorted({0, 84, 99, 143, 200, num_slots - 1})
    print(f"诊断时间片: {key_slots}")

    for slot_idx in key_slots:
        summarize_time_slice(car_time_tensor[slot_idx], hub_idx, inverse_node_map, slot_idx)

    always_unreachable_from_hub = np.ones(num_nodes, dtype=bool)
    always_unreachable_to_hub = np.ones(num_nodes, dtype=bool)

    for slot_idx in key_slots:
        finite_mask = np.isfinite(car_time_tensor[slot_idx])
        always_unreachable_from_hub &= ~finite_mask[hub_idx]
        always_unreachable_to_hub &= ~finite_mask[:, hub_idx]

    always_unreachable_from_hub[hub_idx] = False
    always_unreachable_to_hub[hub_idx] = False

    unreachable_both = np.where(always_unreachable_from_hub & always_unreachable_to_hub)[0]
    sample_both = [inverse_node_map.get(int(idx), str(int(idx))) for idx in unreachable_both[:20]]

    print("\n=== 跨关键时间片汇总 ===")
    print(f"在所有关键时间片中，Hub 始终无法到达的节点数: {int(always_unreachable_from_hub.sum())}")
    print(f"在所有关键时间片中，始终无法回到 Hub 的节点数: {int(always_unreachable_to_hub.sum())}")
    print(f"在所有关键时间片中，与 Hub 始终双向不连通的节点数: {len(unreachable_both)}")
    print(f"始终双向不连通节点样本: {sample_both}")

    print("\n诊断完成：如果 Hub 双向可达节点很少，说明不是 ALNS 本身的问题，而是底层地面时变路网矩阵存在大范围不连通。")


if __name__ == '__main__':
    main()
