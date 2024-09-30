import itertools
import json
import os
import pickle

from matplotlib import pyplot as plt


def load_real_tracks(scene_name):
    """

    :param filename: 
    """
    base_path = f'OutputData/RealTracks/'
    file_extension = '.pkl'
    file_path = f'{base_path}{scene_name}{file_extension}'
    try:
        with open(file_path, 'rb') as f:
            real_tracks = pickle.load(f)
        print(f"RealTracks loaded from {file_path}")
        return real_tracks
    except FileNotFoundError:
        print(f"The file {scene_name}{file_extension} was not found.")
        return None


def load_json_scene_names():
    with open('GroundTruth/scene_names.json', 'rb') as f:
        scene_names = json.load(f)
    return scene_names


def load_json_GroundTruth():
    with open('GroundTruth/ground_truth_data.json', 'rb') as f:
        ground_truth_data = json.load(f)
    return ground_truth_data


def load_json_label_mapping():
    with open('GroundTruth/label_mapping.json', 'rb') as f:
        label_mapping = json.load(f)
    return label_mapping


def load_noLabel_cluster(scene_name):
    pass
def load_input_centroids_data_path(scene_name):
    data_path = f'InputData/WithLabel/LabelCentroids_{scene_name}.pkl'
    print(f'\n\n\nLoading {data_path}')
    return data_path


def get_available_filename(directory, base_filename):
    """Gets the available file name, adding a sequence number to the end if there is a file with the same name"""
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    new_file_path =os.path.join(directory,base_filename)
    while os.path.exists(new_file_path):
        new_filename = f"{filename}_{counter}{ext}"
        new_file_path = os.path.join(directory, new_filename)
        counter += 1
    return new_file_path


def save_noLabel_inputData(scene_name, Centroids, cluster_data):
    
    centroid_dir = 'InputData/NoLabel'
    cluster_dir = 'InputData/NoLabel'

  
    os.makedirs(centroid_dir, exist_ok=True)
    os.makedirs(cluster_dir, exist_ok=True)
    
    centroid_filename = get_available_filename(centroid_dir, f'Centroids_{scene_name}.pkl')
    cluster_filename = get_available_filename(cluster_dir, f'Cluster_{scene_name}.pkl')
    
    with open(centroid_filename, 'wb') as file:
        pickle.dump(Centroids, file)
    print(f"NoLabel Centroids saved to {centroid_filename}")
    
    with open(cluster_filename, 'wb') as file:
        pickle.dump(cluster_data, file)
    print(f"NoLabel cluster_data saved to {cluster_filename}")



def get_in_channels_and_extension(semantic_features):
    extension=''
    
    in_channels = 3+2  
    if semantic_features == ['v'] :
        in_channels = 3+1  
        extension='_v'
    elif semantic_features == ['RCS']:
        in_channels=3+1
        extension='_RCS'
    elif semantic_features == []:
        in_channels = 3  
        extension='_xyz'
    return in_channels,extension

def get_channels_and_extension(semantic_features, withSRR):
    in_channels, extension = get_in_channels_and_extension(semantic_features)
    if not withSRR and semantic_features == ['v', 'RCS']:
        extension = 'noSRR'
    if not withSRR and semantic_features != ['v', 'RCS']:
        raise ValueError(
            f"Parameter conflict: withSRR={withSRR}, semantic_features={semantic_features}. "
            "Please check the parameter settings to ensure there is no conflict between withSRR and semantic_features."
        )
    return in_channels, extension


def plot_data_5d(data_5d,frame_id=None):
    if frame_id is None:
        data_5d = data_5d
    else:
        data_5d=data_5d[frame_id:frame_id+1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in data_5d:
        x = frame[:, 0]  # x
        y = frame[:, 1]  # y
        z = frame[:, 2]  # z
        v = frame[:, 3]  # v

       
        sc = ax.scatter(x, y, z, c=v, cmap='viridis', marker='o')

    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('v')

    

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.show()

def plot_data_cluster(cluster_data,frame_id=None):
    if frame_id is None:
        cluster_data = cluster_data
    else:
        cluster_data=cluster_data[frame_id:frame_id+1]

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = itertools.cycle(plt.cm.tab10.colors)  

    
    for i, groups in enumerate(cluster_data):
        
        for key, group_data in groups.items():
           
            color = next(colors)

            
            x = group_data[:, 0]  # x
            y = group_data[:, 1]  # y
            z = group_data[:, 2]  # z

            
            ax.scatter(x, y, z, color=color, label=f'Group {key}', alpha=0.6, marker='o')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    plt.show()
