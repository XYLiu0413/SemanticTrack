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
        l0_points = points[:, :, 3].unsqueeze(-1)  # Select the fourth channel and keep the shape (B, N, 1)
    elif extension == '_RCS':
        l0_points = points[:, :, 4].unsqueeze(-1)  # Select the 5th channel and keep the shape (B, N, 1)
    elif extension == '_xyz':
        l0_points = None  #
    else:
        l0_points = points[:, :, 3:]  # default
    return l0_points


if __name__ == '__main__':
    with open('E:\Projects\SemanticTrack\DataSets\self\dataset_all.pkl','rb') as f:
        data = pickle.load(f)
    print('load data successful')
