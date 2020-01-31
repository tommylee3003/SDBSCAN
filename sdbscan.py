import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from collections import deque
from scipy.stats import norm

import sdbscan_merge_chain

# from sklearn.preprocessing import PowerTransformer

from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd

from fastkNN import fastkNN


def _build_chain(data_len, graph_csr):
    per = np.empty(len(graph_csr.indices), dtype=np.intp)
    per[:] = graph_csr.indices
    noise = np.zeros(len(graph_csr.indices), dtype=np.intp)
    for i in range(data_len):
        r = i
        while True:
            if per[per[r]] == r:
                per[r] = r
                break
            if (graph_csr.data[i] + 1e-7) / (graph_csr.data[graph_csr.indices[i]] + 1e-7) <= 2 \
                    and not noise[graph_csr.indices[i]]:
                r = per[r]
            else:
                noise[r] = 1
                break
        j = i
        while j != r:
            z = per[j]
            per[j] = r
            j = z

    return per, noise


'''
聚类主函数
输入：core:            核心点集
     neighbors:       核心点领域内点集
     core_map:        序号-核心点映射
     lables:          初始化后的标签数组

输出：聚类结果标签数组

     算法使用变型的深度优先遍历算法，对每个点进行聚类并标注

'''


def _sdbscan_merge_chain(is_core, per_index, neighbors, d_index, labels, radius, noise):
    stack = deque()
    lables_num = 0
    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[per_index[i]] or noise[i]:
            continue

        while True:
            if labels[i] == -1:
                now_core = per_index[i]
                # labels[per_index == now_core] = lables_num
                labels[i] = lables_num
                # nebr = np.unique(np.extract(neighbors_list[0][per_index == now_core] <= radius,
                # neighbors_list[1][per_index == now_core]))
                nebr = neighbors[i]
                for j in range(nebr.shape[0]):
                    v = nebr[j]
                    if labels[v] == -1:
                        if per_index[v] == now_core and not noise[v]:
                            stack.append(v)
                            continue
                        elif is_core[per_index[v]] and not noise[v]:
                            if d_index[i][j] <= radius[v]:
                                stack.append(v)
                            else:
                                continue
                        else:
                            labels[v] = lables_num
            if not stack:
                break
            i = stack.pop()
        lables_num += 1

    return labels



'''
输入：
    X:                 数据集
    min_samples:       回溯每个点的近邻数及聚类半径参数，默认为 20
    min_cluster_size:  最小簇大小，默认为 None
    noise_percent:     先验噪声点比率
    leaf_size:         空间分割树叶结点大小，默认为30

输出：聚类结果标签数组
'''


def sdbscan(X, min_samples=20, min_cluster_size=None, noise_percent=0.0, metric='euclidean',
            leaf_size=30, density_mode='global', silence=True):
    n_neighbor = max(20, 2 * min_samples)
    '''
    model = NearestNeighbors(algorithm='kd_tree', metric=metric, n_neighbors=n_neighbor, leaf_size=leaf_size)
    model.fit(X)
    graph_csr = model.kneighbors_graph(n_neighbors=1, mode='distance')
    nearest_list = model.kneighbors()  # type: np.ndarray
    '''
    if not silence:
        from time import time
        start = time()
    model = fastkNN(n_neighbor, metric=metric, return_distance=True, n_trees=10)
    model.fit(X)
    graph_csr = model.kneighbors_graph(n_neighbors=1)
    nearest_list = model.kneighbors()

    if not silence:
        end = time()
        print('build NN-graph and neighbors search completed in', end-start, 's')
        print('core samples search start')

    chain_index, noise = _build_chain(X.shape[0], graph_csr)
    chain_index = np.asarray(chain_index, dtype=np.intp)
    noise = np.asarray(noise, dtype=np.intp)

    chain_len = pd.value_counts(chain_index[noise == 0], sort=False)
    # core = np.extract(chain_len > 2, chain_len.index)
    # core = np.unique(chain_index)
    # core = np.where(chain_index == np.arange(0, X.shape[0], dtype=np.int64))[0]
    core = np.asarray(chain_len.index, dtype=np.intp)

    if not silence:
        end1 = time()
        print('core samples search completed in', end1-end, 's')
        print('core refine and noise search start')
    '''
    # 链长比大于2的点被标记为噪音
    noise = np.zeros(X.shape[0], dtype=np.intp)
    if check_distance:
        ratio = [(graph_csr.data[i] + 1e-7) / (graph_csr.data[graph_csr.indices[i]] + 1e-7) for i in range(X.shape[0])]
        noise = np.where(np.asarray(ratio) <= 2, noise, 1)
    '''
    if noise_percent > 0:
        # 只计算核心点的密度分布
        if density_mode == 'core_only':
            mean_list = np.mean(nearest_list[0][core, 0:(min_samples - 1)], axis=1)
            mean_list = np.extract(mean_list > 0, mean_list)
            mean_list = np.log(mean_list)
            # mean_list = PowerTransformer().fit_transform(mean_list.reshape(-1, 1))
            mean_r = np.mean(mean_list) + norm.ppf(1 - noise_percent) * np.std(mean_list, ddof=1)
            core = np.extract(mean_list <= mean_r, core)

        # 计算所有点的密度分布
        elif density_mode == 'global':
            mean_list = np.mean(nearest_list[0][:, 0:(min_samples - 1)], axis=1)
            mean_list = np.log(mean_list)
            # mean_list = PowerTransformer().fit_transform(mean_list.reshape(-1, 1))
            mean_r = np.mean(mean_list) + norm.ppf(1 - noise_percent) * np.std(mean_list, ddof=1)
            core = np.extract(mean_list[core] <= mean_r, core)
        else:
            raise ValueError('Wrong density_mode: {core_only}/{global} is acceptable')

    radius = np.zeros(X.shape[0])
    for i in range(core.shape[0]):
        radius[chain_index == core[i]] = (1 + 1 / (1 + np.exp(chain_len.max() - chain_len[core[i]]))) \
                                         * np.mean(nearest_list[0][core[i]][0:(min_samples - 1)])
    neighbors = [np.extract(nearest_list[0][i] <= radius[i], nearest_list[1][i]) for i in range(radius.shape[0])]
    neighbors = np.asarray(neighbors)

    if not silence:
        end2 = time()
        print('core refine and noise search completed in', end2-end1, 's')
        print('clustering start')

    lables = -np.ones(X.shape[0], dtype=np.intp)
    is_core = np.zeros(X.shape[0], dtype=np.intp)
    is_core[core] = 1

    # lables = _sdbscan_merge_chain(is_core, chain_index, neighbors, nearest_list[0], lables, radius, noise)

    lables = sdbscan_merge_chain.sdbscan_merge_chain(is_core,
                                                     neighbors,
                                                     nearest_list[0],
                                                     chain_index,
                                                     lables,
                                                     noise,
                                                     radius)

    if not silence:
        end3 = time()
        print('clustering completed in', end3-end2, 's')
        print('algorithm take', end3-start, 's')

    if min_cluster_size:
        count = Counter(lables)
        for i in np.unique(lables):
            if count.get(i) < min_cluster_size:
                lables = np.where(lables != i, lables, -1)

    return lables


class SDBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self,
                 min_samples=20,
                 min_cluster_size=None,
                 noise_percent=0.0,
                 metric='euclidean',
                 leaf_size=30,
                 density_mode='global',
                 silence=True):
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.noise_percent = noise_percent
        self.metric = metric
        self.leaf_size = leaf_size
        self.density_mode = density_mode
        self.silence = silence

    def fit(self, X):
        clust = sdbscan(X, **self.get_params())
        self.labels_ = clust
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
