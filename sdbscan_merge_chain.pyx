cimport cython
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np

cdef inline void push(vector[np.npy_intp] &stack, np.npy_intp i) except +:
    stack.push_back(i)

@cython.boundscheck(False)
@cython.wraparound(False)

def sdbscan_merge_chain(np.ndarray[np.int64_t, ndim=1, mode='c'] is_core,
                  np.ndarray[object] neighborhoods,
                  np.ndarray[double, ndim=2] neighbor_dist,
                  np.ndarray[np.npy_intp, ndim=1, mode='c'] per_index,
                  np.ndarray[np.npy_intp, ndim=1, mode='c'] labels,
                  np.ndarray[np.npy_intp, ndim=1, mode='c'] noise,
                  np.ndarray[double, ndim=1] radius):

    cdef np.npy_intp i, label_num = 0, v
    cdef vector[np.npy_intp] stack
    cdef np.ndarray[long, ndim=1] nebr_core
    cdef np.ndarray[long, ndim=1] nebr_exp
    cdef np.ndarray[long, ndim=1] subgraph

    for i in range(labels.shape[0]):
        if labels[i] != -1:
            continue
        elif not is_core[per_index[i]]:
            continue
        elif noise[i]:
            continue

        while True:
            if labels[i] == -1:
                labels[i] = label_num
                if is_core[i]:
                    subgraph = np.asarray(np.where(per_index == i)[0], dtype=np.long)
                    labels[subgraph] = label_num
                    core_nebr = neighborhoods[i]
                    exp_nebr = np.unique(np.hstack(neighborhoods[per_index == i]))

                    for j in range(core_nebr.shape[0]):
                        v = core_nebr[j]
                        if is_core[v]:
                            if neighbor_dist[i][j] <= radius[v]:
                                push(stack, v)
                                continue
                            else:
                                continue
                        else:
                            continue

                    for j in range(exp_nebr.shape[0]):
                        v = exp_nebr[j]
                        if labels[v] == -1 and not is_core[per_index[v]]:
                            labels[v] = label_num
                        else:
                            continue

            if stack.size() == 0:
                break
            i = stack.back()
            stack.pop_back()
        label_num += 1
    return labels
