import torch
import numpy as np
from sklearn.neighbors import KDTree
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
try:
    import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
except BaseException:
    print('Failed to import cpp_neighbors, nanoflann kNN is not loaded. Only sklearn kNN is available')



def grid_subsampling(
        points,
        features=None,
        labels=None,
        sampleDl=0.1,
        verbose=0):
    """
    This function comes from https://github.com/HuguesTHOMAS/KPConv
    Copyright Hugues Thomas 2018
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    Input:
        points: (N, 3) matrix of input points
        features: optional (N, d) matrix of features (floating number)
        labels: optional (N,) matrix of integer labels
        sampleDl: parameter defining the size of grid voxels
        verbose: 1 to display
        subsampled points, with features and/or labels depending of the input
    Output: subsampled points
    """


    # method = "voxelcenters" # "barycenters" "voxelcenters"
    method = "barycenters"

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(
            points,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)
    elif (labels is None):
        return cpp_subsampling.compute(
            points,
            features=features,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)
    elif (features is None):
        return cpp_subsampling.compute(
            points,
            classes=labels,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)
    else:
        return cpp_subsampling.compute(
            points,
            features=features,
            classes=labels,
            sampleDl=sampleDl,
            verbose=verbose,
            method=method)


def compute_knn(ref_points, query_points, K, dilated_rate=1, method='sklearn'):
    """
    Compute KNN
    Input:
        ref_points: reference points (MxD)
        query_points: query points (NxD)
        K: the amount of neighbors for each point
        dilated_rate: If set to larger than 1, then select more neighbors and then choose from them
        (Engelmann et al. Dilated Point Convolutions: On the Receptive Field Size of Point Convolutions on 3D Point Clouds. ICRA 2020)
        method: Choose between two approaches: Scikit-Learn ('sklearn') or nanoflann ('nanoflann'). In general nanoflann should be faster, but sklearn is more stable
    Output:
        neighbors_idx: for each query point, its K nearest neighbors among the reference points (N x K)
    """
    num_ref_points = ref_points.shape[0]

    if num_ref_points < K or num_ref_points < dilated_rate * K:
        num_query_points = query_points.shape[0]
        inds = np.random.choice(
            num_ref_points, (num_query_points, K)).astype(
            np.int32)

        return inds
    if method == 'sklearn':
        kdt = KDTree(ref_points)
        neighbors_idx = kdt.query(
            query_points,
            k=K * dilated_rate,
            return_distance=False)
    elif method == 'nanoflann':
        neighbors_idx = batch_neighbors(
            query_points, ref_points, [
                query_points.shape[0]], [num_ref_points], K * dilated_rate)
    else:
        raise Exception('compute_knn: unsupported knn algorithm')
    if dilated_rate > 1:
        neighbors_idx = np.array(
            neighbors_idx[:, ::dilated_rate], dtype=np.int32)

    return neighbors_idx    


def subsample_and_knn(
        coord,
        norm,
        grid_size=[0.1],
        K_self=16,
        K_forward=16,
        K_propagate=16):
    """
    Perform grid subsampling and compute kNN at each subsampling level
    Input:
        coord: N x 3 coordinates
        norm: N x 3 surface normals
        grid_size: all the downsampling levels (in cm) you want to use, e.g. [0.05, 0.1, 0.2, 0.4, 0.8]
        K_self: number of neighbors within each downsampling level
        K_forward: number of neighbors from one downsampling level to the next one (with less points), used in the downsampling PointConvs in the encoder
        K_propagate: number of neighbors from one downsampling level to the previous one (with more points), used in the upsampling PointConvs in the decoder
    Outputs:
        point_list: list of length len(grid_size)
        nei_forward_list: downsampling kNN neighbors (K_forward neighbors for each point)
        nei_propagate_list: upsampling kNN neighbors (K_propagate neighbors for each point)
        nei_self_list: kNN neighbors between the same layer
        norm_list: list of surface normals averaged within each voxel at each grid_size
    """
    point_list, norm_list = [], []
    nei_forward_list, nei_propagate_list, nei_self_list = [], [], []
    for j, grid_s in enumerate(grid_size):
        if j == 0:
            # sub_point, sub_norm = coord.astype(
            #     np.float32), norm.astype(
            #     np.float32)
            
            sub_point = coord.astype(np.float32)
            sub_norm = None

            point_list.append(sub_point)
            norm_list.append(sub_norm)
            # compute edges
            nself = compute_knn(sub_point, sub_point, K_self[j])
            nei_self_list.append(nself)

        else:
            sub_point, sub_norm = grid_subsampling(
                points=point_list[-1], features=None, sampleDl=grid_s)

            if sub_point.shape[0] <= K_self[j]:
                sub_point, sub_norm = point_list[-1], None

            # compute edges, nforward is for downsampling, npropagate is for upsampling,
            # nself is for normal PointConv layers (between the same set of
            # points)
            nforward = compute_knn(point_list[-1], sub_point, K_forward[j])
            npropagate = compute_knn(sub_point, point_list[-1], K_propagate[j])
            nself = compute_knn(sub_point, sub_point, K_self[j])
            # point_list is a list with len(grid_size) length, each item is a numpy array
            # of num_points x dimensionality

            point_list.append(sub_point)
            norm_list.append(sub_norm)
            nei_forward_list.append(nforward)
            nei_propagate_list.append(npropagate)
            nei_self_list.append(nself)

    return point_list, nei_forward_list, nei_propagate_list, nei_self_list, norm_list


def batch_neighbors(queries, supports, q_batches, s_batches, K):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param K: long
    :return: neighbors indices
    """

    return cpp_neighbors.batch_kquery(
        queries, supports, q_batches, s_batches, K=int(K))