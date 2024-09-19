import os

import numpy as np
import scipy.io as sio
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN

from utils.common import load_json_GroundTruth, load_json_scene_names, save_noLabel_inputData, \
    get_in_channels_and_extension, plot_data_5d, plot_data_cluster


def frame_overlay(path):
    """
    将每组五帧连续帧组合成一个数据集。

    参数:
    path (str): 包含雷达数据的 .mat 文件路径。

    返回:
    tuple: 包含以下内容的元组：
        - data_5d (list of np.ndarray): 每个元素是一个 numpy 数组，合并了五帧数据。
        - filename (str): 输入 .mat 文件的基本文件名。
    """
    filename = os.path.splitext(os.path.basename(path))[0]
    data = sio.loadmat(path)
    xyz_all = data['xyz_all'][0]  # 假设 xyz_all 是 MATLAB 中的一维单元数组
    if len(xyz_all) > 5:
        data_5d = []
        for i in range(0, len(xyz_all) - len(xyz_all) % 5, 5):
            xyzv = np.vstack([xyz_all[i], xyz_all[i + 1], xyz_all[i + 2], xyz_all[i + 3], xyz_all[i + 4]])
            selected_columns = np.hstack((xyzv[:, :4], 10 * np.log10(xyzv[:, 5:6])))  # 提取第1-4列和第6列,并且第六列取db为单位
            data_5d.append(selected_columns)
    else:
        raise ValueError('点云帧数小于5，无法进行叠加！')
    # df=pd.DataFrame(data_5d,colunms=['x', 'y', 'z', 'v', 'RCS'])
    return data_5d, filename


def spatial_cluster(data_5d, filename=None):
    data_5d_DBSCAN = []
    Centroids = []
    for i, xyzvp in enumerate(data_5d):
        centroid = np.empty((0, 6))
        k = 0
        # if i==29:
        moveID = (xyzvp[:, 2] >= -1) & (abs(xyzvp[:, 3]) > 0.1)
        X = xyzvp[moveID]
        # DBSCAN 聚类
        if X.size > 0:
            clustering = DBSCAN(eps=0.5, min_samples=15).fit(X[:, :3])
            labels = clustering.labels_[clustering.core_sample_indices_]
            groups = {}
            for label in np.unique(labels):
                # 使用核心样本索引和标签索引筛选属于同一标签的数据
                tmp = X[clustering.core_sample_indices_]
                group_indices = tmp[labels == label]
                if len(group_indices) > 15:
                    groups[label] = group_indices
                    xyzv = np.mean(group_indices[:, :4], axis=0)
                    xyzv_key = np.append(xyzv, [label, 0])
                    centroid = np.vstack((centroid, xyzv_key))
            Centroids.append(centroid)
            data_5d_DBSCAN.append(groups)
    return Centroids, data_5d_DBSCAN


def KDE(data_5d_DBSCAN, semantic_features=None, filename=None):
    if semantic_features is None:
        semantic_features = ['v', 'RCS']
    data_5d_KDE = []
    Centroids = []
    for i, groups in enumerate(data_5d_DBSCAN):
        groups_KDE = {}
        centroid = np.empty((0, 6))
        k = 0
        for key in groups.keys():
            if semantic_features == ['v']:
                v_p_data = groups[key][:, 3]
            elif semantic_features == ['RCS']:
                v_p_data = groups[key][:, 4]
            else:
                v_p_data = groups[key][:, [3, 4]]
            noise = 0.0001 * np.random.normal(size=v_p_data.shape)
            v_p_data = v_p_data + noise
            # 计算原始数据的核密度估计
            original_kde = gaussian_kde(v_p_data.T)
            # 计算核密度估计和每个数据点的密度
            densities = original_kde(v_p_data.T)
            density_threshold = np.percentile(densities, 25)
            filtered_data = groups[key][densities > density_threshold]
            if len(filtered_data) > 15:
                groups_KDE[key] = filtered_data
                xyzv = np.mean(filtered_data[:, :4], axis=0)
                xyzv_key = np.append(xyzv, [k, 0])
                k += 1
                centroid = np.vstack((centroid, xyzv_key))
        Centroids.append(centroid)
        data_5d_KDE.append(groups_KDE)
    return Centroids, data_5d_KDE


def srr(withSRR=True, semantic_features=None,save=True,frame_id=None):
    if semantic_features is None:
        semantic_features = ['v', 'RCS']
    scene_names = load_json_scene_names()
    ground_truth_data = load_json_GroundTruth()
    # scene_name=scene_names[1]
    for scene_name in scene_names[1:4]:
        scene_data = ground_truth_data[scene_name]
        path = scene_data['mat_path']
        data_5d, filename = frame_overlay(path)
        plot_data_5d(data_5d,frame_id)
        Centroids, data_5d_DBSCAN = spatial_cluster(data_5d)
        _, extension = get_in_channels_and_extension(semantic_features)
        if withSRR and semantic_features !=[]:
            Centroids, data_5d_KDE = KDE(data_5d_DBSCAN, semantic_features)
            plot_data_cluster(data_5d_KDE,frame_id)
            scene_name = scene_name + extension
            if save:
                save_noLabel_inputData(scene_name, Centroids, data_5d_KDE)
        elif not withSRR:
            extension='noSRR'
            scene_name = scene_name + extension
            if save:
                save_noLabel_inputData(scene_name, Centroids, data_5d_DBSCAN)
        elif semantic_features ==[]:
            extension = '_xyz'
            scene_name = scene_name + extension
            if save:
                save_noLabel_inputData(scene_name, Centroids, data_5d_DBSCAN)





if __name__ == '__main__':
    # srr(withSRR=True,semantic_features=['v'])
    srr(withSRR=True, semantic_features=None, save=False, frame_id=8)

