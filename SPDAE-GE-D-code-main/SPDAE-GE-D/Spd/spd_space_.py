import json
import numpy as np
from scipy.linalg import svd
from scipy.linalg import eigh
from pathlib import Path
class GrassmannSubSpace:
    def __init__(self):
        pass

    def orthogonal_subspace(self, data):
        D = []
        for d in data:
            q, _ = np.linalg.qr(d)
            D.append(q)
        return np.array(D)

    def compute_subspace(self, data, p=10):
        sub = []
        for d in data:
            U, _, _ = svd(np.dot(d, d.T))
            sub.append(U[:, :p])
        return np.array(sub)

class GrassmannDimensionality:
    def __init__(self, ratio=0.95):
        self.ratio = ratio
    def stack(self, data):
        s = np.zeros((data[0].shape[0], data[0].shape[0])).astype(np.float32)
        for d in data:
            s += np.dot(d, d.T)
        s /= data.shape[0]
        return s

    def determine_dimensionality(self, data):
        s = self.stack(data)
        values = eigh(s, eigvals_only=True)[::-1]
        total_variance = np.sum(values)
        explained_variance_ratio = values / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        self.components = int(np.argmax(cumulative_variance_ratio >= self.ratio) + 1)
        return self.components

    def save_low_dimensions(self, data_name):
        root = Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]
        leaf = ["DATA", "GRASSMANN", "Grassmann_data_paras.json"]
        root = list(root) + leaf
        json_path = "/".join(root)
        with open(json_path, 'r', encoding='utf-8') as paras:
            grassmann_paras = json.load(paras)
        paras.close()
        if grassmann_paras["low_dimensions"].get(data_name) is None:
            grassmann_paras["low_dimensions"][data_name] = dict()
        if not isinstance(grassmann_paras["low_dimensions"][data_name], dict):
            grassmann_paras["low_dimensions"][data_name] = dict()
        grassmann_paras["low_dimensions"][data_name][self.ratio] = self.components
        with open(json_path, 'w') as paras:
            json.dump(grassmann_paras, paras, indent=4)
        paras.close()
