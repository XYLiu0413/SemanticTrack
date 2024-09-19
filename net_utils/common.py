import torch


def gather_points(points, indexes):
    '''

    :param points: shape=(B, N, C)
    :param indexes: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    indexes_shape = list(indexes.shape)
    indexes_shape[1:] = [1] * len(indexes_shape[1:])
    repeat_shape = list(indexes.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(indexes_shape).repeat(repeat_shape)
    return points[batchlists, indexes, :]


def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: indexes: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    indexes = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = indexes
        cur_point = xyz[batchlists, indexes, :]  # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        indexes = torch.max(dists, dim=1)[1]
    return centroids


def ball_query(xyz, new_xyz, radius, K):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_indexes = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_indexes[dists > radius] = N
    grouped_indexes = torch.sort(grouped_indexes, dim=-1)[0][:, :, :K]
    grouped_min_indexes = grouped_indexes[:, :, 0:1].repeat(1, 1, K)
    grouped_indexes[grouped_indexes == N] = grouped_min_indexes[grouped_indexes == N]
    return grouped_indexes


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def three_interpolate(xyz1, xyz2, points2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :param points2: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = points2.shape
    dists, indexes = three_nn(xyz1, xyz2)
    inverse_dists = 1.0 / (dists + 1e-8)
    weight = inverse_dists / torch.sum(inverse_dists, dim=-1, keepdim=True)  # shape=(B, N1, 3)
    weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, C2)
    interpolated_points = gather_points(points2, indexes)  # shape=(B, N1, 3, C2)
    interpolated_points = torch.sum(weight * interpolated_points, dim=2)
    return interpolated_points


def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), indexes: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, indexes = torch.sort(dists, dim=-1)
    dists, indexes = dists[:, :, :3], indexes[:, :, :3]
    return dists, indexes
