import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from net_utils.IoU import compute_metrics
from net_utils.IoU import integral_label
from net_utils.IoU import plot_confusion_matrix
from net_utils.common import ball_query
from net_utils.common import fps
from net_utils.common import gather_points
from net_utils.common import three_interpolate
from net_utils.utils import get_weight_path, get_semantic_points
from utils.common import load_json_scene_names, load_json_GroundTruth, load_json_label_mapping, \
    get_in_channels_and_extension, get_available_filename


class PointCloudData(Dataset):
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]

    @staticmethod
    def combine_datasets(dataset1, dataset2):
        with open(dataset1, 'rb') as file:
            Single_vehicle = pickle.load(file)
        with open(dataset2, 'rb') as file:
            Single_person_straight = pickle.load(file)
        combined = Single_vehicle + Single_person_straight
        with open('DataSets/self/combined.pkl', 'wb') as file:
            pickle.dump(combined, file)

    @staticmethod
    def load_from_pkl(file_paths, num_points=1, split_ratio=0.8):

        points_list = []
        labels_list = []
        current_points = []
        current_labels = []
        label_mapping = {'pedestrian': 0, 'vehicle': 1, 'ghost': 2}

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            for point, label in data:
                if len(current_points) < num_points:
                    current_points.append(point)
                    current_labels.append(label_mapping[label])
                if len(current_points) == num_points:
                    points_list.append(np.array(current_points))
                    labels_list.append(np.array(current_labels))
                    current_points = []
                    current_labels = []

        # The data is divided into training sets and validation sets
        dataset_size = len(points_list)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(split_ratio * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]

        train_points = [points_list[i] for i in train_indices]
        train_labels = [labels_list[i] for i in train_indices]
        val_points = [points_list[i] for i in val_indices]
        val_labels = [labels_list[i] for i in val_indices]

        return PointCloudData(train_points, train_labels), PointCloudData(val_points, val_labels)

    @staticmethod
    def test_data(cluster, num_points=1):
        label_mapping = {'pedestrian': 0, 'vehicle': 1, 'ghost': 2}
        points = cluster[0]
        label = label_mapping[cluster[1]]
        labels = [label] * len(cluster[0])
        points_list = []
        labels_list = []
        current_points = []
        current_labels = []
        for point, label in zip(points, labels):
            if len(current_points) < num_points:
                current_points.append(point)
                current_labels.append(label)
            if len(current_points) == num_points:
                points_list.append(np.array(current_points))
                labels_list.append(np.array(current_labels))
                current_points = []
                current_labels = []
        return points_list, labels_list

    @staticmethod
    def prediction_data(points, num_points=1):
        labels = [0] * len(points)
        points_list = []
        labels_list = []
        current_points = []
        current_labels = []
        for point, label in zip(points, labels):
            if len(current_points) < num_points:
                current_points.append(point)
                current_labels.append(label)
            if len(current_points) == num_points:
                points_list.append(np.array(current_points))
                labels_list.append(np.array(current_labels))
                current_points = []
                current_labels = []
        return points_list, labels_list


class PointNet_SA_Module_MSG(nn.Module):
    def __init__(self, M, radiuses, Ks, in_channels, mlps, bn=True, pooling='max', use_xyz=True):
        super(PointNet_SA_Module_MSG, self).__init__()
        self.M = M
        self.radiuses = radiuses
        self.Ks = Ks
        self.in_channels = in_channels
        self.mlps = mlps
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self.backbones = nn.ModuleList()
        for j in range(len(mlps)):
            mlp = mlps[j]
            backbone = nn.Sequential()
            in_channels = self.in_channels
            for i, out_channels in enumerate(mlp):
                backbone.add_module('Conv{}_{}'.format(j, i),
                                    nn.Conv2d(in_channels, out_channels, 1,
                                              stride=1, padding=0, bias=False))
                if bn:
                    backbone.add_module('Bn{}_{}'.format(j, i),
                                        nn.BatchNorm2d(out_channels))
                backbone.add_module('Relu{}_{}'.format(j, i), nn.ReLU())
                in_channels = out_channels
            self.backbones.append(backbone)

    def forward(self, xyz, points):
        new_xyz = gather_points(xyz, fps(xyz, self.M))  # (B, M, 3)
        new_points_all = []
        for i in range(len(self.mlps)):
            radius = self.radiuses[i]
            K = self.Ks[i]
            grouped_indexes = ball_query(xyz, new_xyz, radius, K)  # (B, M, K)
            grouped_xyz = gather_points(xyz, grouped_indexes)  # (B, M, K, 3)
            grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
            if points is not None:
                grouped_points = gather_points(points, grouped_indexes)  # (B, M, K, C)
                if self.use_xyz:
                    new_points = torch.cat(
                        (grouped_xyz.float(), grouped_points.float()),  # (B, M, K, C+3)
                        dim=-1)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
            new_points = self.backbones[i](new_points.permute(0, 3, 2, 1).contiguous())  # (B, C', M, K)
            if self.pooling == 'avg':
                new_points = torch.mean(new_points, dim=2)
            else:
                new_points = torch.max(new_points, dim=2)[0]  # (B, C', M)
            new_points = new_points.permute(0, 2, 1).contiguous()  # (B, M, C')
            new_points_all.append(new_points)
        return new_xyz, torch.cat(new_points_all, dim=-1)


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(PointNet_FP_Module, self).__init__()
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels

    def forward(self, xyz1, xyz2, points1, points2):
        '''

        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''
        B, N1, C1 = points1.shape
        _, N2, C2 = points2.shape
        if N2 == 1:
            interpolated_points = points2.repeat(1, N1, 1)
        else:
            interpolated_points = three_interpolate(xyz1, xyz2, points2)
        cat_interpolated_points = torch.cat([interpolated_points, points1], dim=-1).permute(0, 2, 1).contiguous()
        new_points = torch.squeeze(self.backbone(torch.unsqueeze(cat_interpolated_points, -1)), dim=-1)
        return new_points.permute(0, 2, 1).contiguous()


class pointnet2_seg_msg_radar(nn.Module):
    def __init__(self, in_channels, nclasses):
        super(pointnet2_seg_msg_radar, self).__init__()
        self.pt_sa1 = PointNet_SA_Module_MSG(M=1, radiuses=[0.5, 0.5], Ks=[1, 1], in_channels=in_channels,
                                             mlps=[[32, 32, 64], [64, 64, 128]])
        self.pt_sa2 = PointNet_SA_Module_MSG(M=1, radiuses=[0.5, 0.5], Ks=[1, 1], in_channels=64 + 128 + 3,
                                             mlps=[[32, 32, 64], [64, 64, 128]])
        self.pt_sa3 = PointNet_SA_Module_MSG(M=1, radiuses=[0.5, 0.5], Ks=[1, 1], in_channels=64 + 128 + 3,
                                             mlps=[[64, 64, 128], [64, 64, 128]])

        self.pt_fp1 = PointNet_FP_Module(in_channels=128 + 128 + 64 + 128, mlp=[256, 256], bn=True)
        self.pt_fp2 = PointNet_FP_Module(in_channels=128 + 128 + 64 + 128, mlp=[128, 128], bn=True)
        self.pt_fp3 = PointNet_FP_Module(in_channels=128 + in_channels, mlp=[128, 128, 128], bn=True)

        self.conv1 = nn.Conv1d(128, 256, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, 128, 1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.cls = nn.Conv1d(128, nclasses, 1, stride=1)

    def forward(self, l0_xyz, l0_points):
        # print(f'l0_points:{l0_points.shape}')
        l1_xyz, l1_points = self.pt_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.pt_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.pt_sa3(l2_xyz, l2_points)

        l2_points = self.pt_fp1(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.pt_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if l0_points == None:
            l0_points = self.pt_fp3(l0_xyz, l1_xyz, l0_xyz, l1_points)
        else:
            l0_points = self.pt_fp3(l0_xyz, l1_xyz, torch.cat([l0_points, l0_xyz], dim=-1), l1_points)

        net = l0_points.permute(0, 2, 1).contiguous()
        net = self.dropout1(F.relu(self.bn1(self.conv1(net))))
        net = self.dropout2(F.relu(self.bn2(self.conv2(net))))
        net = self.cls(net)

        return net


class seg_loss(nn.Module):
    def __init__(self):
        super(seg_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, label):
        '''

        :param prediction: shape=(B, N, C)
        :param label: shape=(B, N)
        :return:
        '''
        loss = self.loss(prediction, label)
        return loss


def evaluate_seg(dataset, checkpoint, batch_size, nclasses, dims, if_save=False):
    print('Loading..')
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = pointnet2_seg_msg_radar(in_channels=dims, nclasses=nclasses)  # 3 calsses
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Dataset: {}, Evaluating..".format(len(dataset)))
    loss_func = seg_loss()
    losses, predictions, labels = [], [], []
    for data, label in tqdm(test_loader):
        data = data.float()
        label = label.long()
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            prediction = model(xyz, points)
            loss = loss_func(prediction, label)
            prediction = torch.max(prediction, dim=1)[1].cpu().detach().numpy()
            predictions.append(prediction)
            losses.append(loss.item())
            labels.append(label.cpu())
    iou, acc, conf_matrix = compute_metrics(np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0),
                                            {'Scenes': [0, 1, 2]})
    plot_confusion_matrix(conf_matrix, ['pedestrian', 'vehicle', 'ghost'], checkpoint, if_save)
    print("Weighed Acc: {:.4f}".format(acc))
    print("Weighed Average IoU: {:.4f}".format(iou))
    print('Mean Loss: {:.4f}'.format(np.mean(losses)))
    print('=' * 40)
    print("Evaluating completed !")


def integral(dataset, checkpoint, batch_size, nclasses, dims, extension=''):
    """
    计算整体的语义标签
    """
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = pointnet2_seg_msg_radar(in_channels=dims, nclasses=nclasses)  
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Dataset: {}, Evaluating..".format(len(dataset)))
    loss_func = seg_loss()
    losses, predictions, labels = [], [], []
    for data, label in tqdm(test_loader):
        data = data.float()
        label = label.long()
        xyz = data[:, :, :3]
        points = get_semantic_points(data, extension)
        with torch.no_grad():
            prediction = model(xyz, points)
            loss = loss_func(prediction, label)
            prediction = torch.max(prediction, dim=1)[1].cpu().detach().numpy()
            predictions.append(prediction)
            losses.append(loss.item())
            labels.append(label.cpu())
    groups_label = integral_label(np.concatenate(predictions, axis=0))
    return groups_label


def prediction(semantic_features=None, withSRR=True):
    """
    prediction code
    :return:
    """
    scene_names = load_json_scene_names()
    label_mapping = load_json_label_mapping()
    ground_truth_data = load_json_GroundTruth()
    if semantic_features is None:
        semantic_features = ['v', 'RCS']  
    in_channels, extension = get_in_channels_and_extension(semantic_features)
    if in_channels == 5:
        if withSRR:
            extension = ''
        else:
            extension = 'noSRR'
    for scene_name in scene_names[1:4]:
        # scene_name=scene_names[1]
        if semantic_features == ['v', 'RCS']:
            weight_path = ground_truth_data[scene_name]['weight_path']
        else:
            weight_path = get_weight_path(semantic_features)
        with open(f'InputData/NoLabel/cluster_{scene_name}{extension}.pkl', 'rb') as file:
            cluster_data = pickle.load(file)
        SemanticClusters_frames = []
        for frame in cluster_data:
            SemanticClusters = []
            for _, data in frame.items():
                ped1points, labels_list = PointCloudData.prediction_data(data)
                pedestrian = PointCloudData(ped1points, labels_list)
                groups_label = integral(pedestrian, weight_path, 10, 3, dims=in_channels, extension=extension)
                SemanticClusters.append((data, groups_label))
            SemanticClusters_frames.append(SemanticClusters)

        WithLabel_dir = 'InputData/WithLabel'
        os.makedirs(WithLabel_dir, exist_ok=True)  
        base_filename = f'LabelCluster_{scene_name}{extension}.pkl'
        file_path = get_available_filename(WithLabel_dir, base_filename)
        with open(file_path, 'wb') as file:
            pickle.dump(SemanticClusters_frames, file)
        print(f'{file_path} saved successfully!')

        Centroids = []
        for frame in SemanticClusters_frames:
            centroid = np.empty((0, 7))
            for i, cluster in enumerate(frame):
                tmp = cluster[0]
                # filtered_data = np.concatenate(tmp, axis=0)
                label = label_mapping[cluster[1]]
                xyzv = np.mean(tmp[:, :4], axis=0)
                xyzv_key = np.append(xyzv, [i, 0, label])
                centroid = np.vstack((centroid, xyzv_key))
            Centroids.append(centroid)

        WithLabel_dir = 'InputData/WithLabel'
        os.makedirs(WithLabel_dir, exist_ok=True) 
        base_filename = f'LabelCentroids_{scene_name}{extension}.pkl'
        file_path = get_available_filename(WithLabel_dir, base_filename)
        with open(file_path, 'wb') as file:
            pickle.dump(Centroids, file)
        print(f'{file_path} saved successfully!')


def train(semantic_features=None, n_classes=3):
    if semantic_features is None:
        semantic_features = ['v', 'RCS']  
    in_channels, extension = get_in_channels_and_extension(semantic_features)
    # load data
    file_paths = ['DataSets/self/dataset_all.pkl']
    train_dataset, val_dataset = PointCloudData.load_from_pkl(file_paths)
    dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=0)
    model = pointnet2_seg_msg_radar(in_channels=in_channels, nclasses=n_classes)  # 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,  # lr (learning rate)
        betas=(0.9, 0.999),
        eps=1e-08,  
        weight_decay=1e-4  
    )
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    for epoch in range(num_epochs):
        for points, labels in dataloader:
            points = points.float()  
            labels = labels.long()  
            optimizer.zero_grad()
            l0_xyz = points[:, :, :3]
            l0_points = get_semantic_points(points, extension)
            outputs = model(l0_xyz, l0_points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')  # , Validation Accuracy: {accuracy}%'

    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)  
    base_filename = f'pointnet2_model{extension}.pth'
    file_path = get_available_filename(weights_dir, base_filename)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


if __name__ == '__main__':
    with open('GroundTruth/scene_names.json', 'rb') as f:
        scene_names = json.load(f)
    with open('GroundTruth/label_mapping.json', 'rb') as f:
        label_mapping = json.load(f)
    with open('GroundTruth/ground_truth_data.json', 'rb') as f:
        ground_truth_data = json.load(f)

    for scene_name in scene_names[1:4]:
        # scene_name=scene_names[1]
        weight_path = ground_truth_data[scene_name]['weight_path']
        with open(f'InputData/NoLabel/cluster_{scene_name}.pkl', 'rb') as file:
            cluster_Multiperson_vehicle = pickle.load(file)
        SemanticClusters_frames = []
        for frame in cluster_Multiperson_vehicle:
            SemanticClusters = []
            for _, data in frame.items():
                ped1points, labels_list = PointCloudData.prediction_data(data)
                pedestrian = PointCloudData(ped1points, labels_list)
                groups_label = integral(pedestrian, weight_path, 10, 3, 3 + 2)
                SemanticClusters.append((data, groups_label))
            SemanticClusters_frames.append(SemanticClusters)
        with open(f'InputData/WithLabel/LabelCluster_{scene_name}.pkl', 'wb') as file:
            pickle.dump(SemanticClusters_frames, file)
        print('SemanticClusters saved successfully!')

        Centroids = []
        for frame in SemanticClusters_frames:
            centroid = np.empty((0, 7))
            for i, cluster in enumerate(frame):
                tmp = cluster[0]
                # filtered_data = np.concatenate(tmp, axis=0)
                label = label_mapping[cluster[1]]
                xyzv = np.mean(tmp[:, :4], axis=0)
                xyzv_key = np.append(xyzv, [i, 0, label])
                centroid = np.vstack((centroid, xyzv_key))
            Centroids.append(centroid)
        with open(f'InputData/WithLabel/LabelCentroids_{scene_name}.pkl', 'wb') as file:
            pickle.dump(Centroids, file)
