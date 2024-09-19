import pickle


def load_train_data():
    pass

def get_weight_path(semantic_features):
    if semantic_features == ['v']:
        weight_path = 'weights/pointnet2_model_v.pth'
    elif semantic_features == ['RCS']:
        weight_path = 'weights/pointnet2_model_RCS.pth'
    elif semantic_features == []:
        weight_path = 'weights/pointnet2_model_xyz.pth'
    return weight_path


def get_semantic_points(points, extension):
    if extension == '_v':
        l0_points = points[:, :, 3].unsqueeze(-1)  # 选择第4个通道，并保持形状为 (B, N, 1)
    elif extension == '_RCS':
        l0_points = points[:, :, 4].unsqueeze(-1)  # 选择第5个通道，并保持形状为 (B, N, 1)
    elif extension == '_xyz':
        l0_points = None  # 如果扩展名为 '_xyz'，则不选择任何特征
    else:
        l0_points = points[:, :, 3:]  # 默认情况下，选择从第4个通道开始的所有特征
    return l0_points


if __name__ == '__main__':
    with open('E:\Projects\SemanticTrack\DataSets\self\dataset_all.pkl','rb') as f:
        data = pickle.load(f)
    print('load data successful')
