import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd


class StaticMapEnvironment:
    def __init__(self, net_file, traffic_file, cache_file):
        self.net_file = net_file
        self.traffic_file = traffic_file
        self.cache_file = cache_file
        self._cache_data = None
        self._net_df = None
        self._avg_speed_map = None

    def load(self):
        if self._cache_data is None:
            with open(self.cache_file, 'rb') as f:
                self._cache_data = pickle.load(f)
        return self

    def _ensure_loaded(self):
        if self._cache_data is None:
            self.load()

    def _load_net_df(self):
        if self._net_df is None:
            net_df = pd.read_csv(self.net_file)
            net_df['link_id'] = net_df['link_id'].astype(str)
            net_df['start_node_id'] = net_df['start_node_id'].astype(str)
            net_df['end_node_id'] = net_df['end_node_id'].astype(str)
            net_df['base_speed'] = pd.to_numeric(net_df['base_speed'], errors='coerce').fillna(40.0)
            self._net_df = net_df
        return self._net_df

    def _load_avg_speed_map(self):
        if self._avg_speed_map is not None:
            return self._avg_speed_map

        traffic_map = {}
        chunk_size = 200_000
        for chunk in pd.read_csv(
            self.traffic_file,
            usecols=['link_id', 'avg_speed'],
            chunksize=chunk_size,
            dtype={'link_id': str, 'avg_speed': float},
        ):
            for _, row in chunk.iterrows():
                lid = str(row['link_id'])
                spd = float(row['avg_speed'])
                if lid not in traffic_map:
                    traffic_map[lid] = []
                traffic_map[lid].append(spd)

        self._avg_speed_map = {lid: np.mean(speeds) for lid, speeds in traffic_map.items()}
        return self._avg_speed_map

    def get_cache_data(self):
        self._ensure_loaded()
        return self._cache_data

    def get_node_map(self):
        self._ensure_loaded()
        return self._cache_data['node_map']

    def get_idx_to_id(self):
        self._ensure_loaded()
        return self._cache_data['idx_to_id']

    def get_coords_array(self):
        self._ensure_loaded()
        return self._cache_data['coords_array']

    def get_risk_array(self):
        self._ensure_loaded()
        return self._cache_data['risk_array']

    def get_static_distance_matrix(self):
        self._ensure_loaded()
        return self._cache_data['dist_matrix_car']

    def get_static_uav_distance_matrix(self):
        self._ensure_loaded()
        return self._cache_data['dist_matrix_uav']

    def get_topology_bundle(self):
        self._ensure_loaded()
        return {
            'node_map': self._cache_data['node_map'],
            'idx_to_id': self._cache_data['idx_to_id'],
            'coords_array': self._cache_data['coords_array'],
            'dist_matrix_car': self._cache_data['dist_matrix_car'],
            'dist_matrix_uav': self._cache_data['dist_matrix_uav'],
            'risk_array': self._cache_data['risk_array'],
        }

    def compute_risk_array(self):
        self._ensure_loaded()
        net_df = self._load_net_df()
        avg_speed_map = self._load_avg_speed_map()
        node_map = self._cache_data['node_map']
        idx_to_id = self._cache_data['idx_to_id']
        num_nodes = len(node_map)

        link_risk = {}
        for _, row in net_df.iterrows():
            lid = str(row['link_id'])
            base_spd = float(row['base_speed'])
            avg_spd = avg_speed_map.get(lid, base_spd)
            loss_rate = max(0.0, (base_spd - avg_spd) / base_spd)
            flow_proxy = 1.0 / max(5.0, avg_spd)
            link_risk[lid] = loss_rate * flow_proxy

        node_risk_sum = defaultdict(float)
        node_risk_count = defaultdict(int)
        for _, row in net_df.iterrows():
            lid = str(row['link_id'])
            r = link_risk.get(lid, 0.0)
            for nid in [str(row['start_node_id']), str(row['end_node_id'])]:
                if nid in node_map:
                    node_risk_sum[nid] += r
                    node_risk_count[nid] += 1

        risk_raw = np.zeros(num_nodes)
        for idx in range(num_nodes):
            nid = idx_to_id[idx]
            cnt = node_risk_count.get(nid, 0)
            if cnt > 0:
                risk_raw[idx] = node_risk_sum[nid] / cnt

        r_min, r_max = risk_raw.min(), risk_raw.max()
        if r_max > r_min:
            risk_normalized = (risk_raw - r_min) / (r_max - r_min) * 10.0
        else:
            risk_normalized = np.ones(num_nodes) * 5.0

        return risk_normalized

    def rebuild_risk_array(self, write_cache=True, make_backup=True):
        self._ensure_loaded()
        risk_array = self.compute_risk_array()
        self._cache_data['risk_array'] = risk_array

        if write_cache:
            backup_path = self.cache_file.replace('.pkl', '_backup_before_rebuild.pkl')
            if make_backup and not os.path.exists(backup_path):
                import shutil
                shutil.copy2(self.cache_file, backup_path)

            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache_data, f)

        return risk_array
