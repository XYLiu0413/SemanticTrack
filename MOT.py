from TrackClasses.TracksManagement import *

from utils.common import load_json_scene_names, load_json_label_mapping, load_json_GroundTruth, \
    load_input_centroids_data_path, get_in_channels_and_extension, get_channels_and_extension

scene_names = load_json_scene_names()
label_mapping = load_json_label_mapping()
ground_truth_data = load_json_GroundTruth()



def mot(semantic_features=None, WithSemantic=True, withSRR=True,plot_vicon=False,plot_ghost=False):
    if semantic_features is None:
        semantic_features = ['v', 'RCS']
    in_channels, extension = get_channels_and_extension(semantic_features=semantic_features,withSRR=withSRR)
    for scene_name in scene_names[1:-1]:
        
        scene_name = scene_name + extension #+ '_1'
        input_data_path = load_input_centroids_data_path(scene_name=scene_name)
        Scene = TracksModule(name=scene_name,extension=extension)
        Scene.load_data(input_data_path)
        Scene.initial_tracks(Scene.Centroids[0])
        Scene.mot_process()
        Scene.report(WithSemantic=WithSemantic)
        # Scene.plot_realtracks()
        # Scene.save_real_tracks()
        # Scene.save_ghost_tracks()
        # Scene.save_none_tracks()
        # Scene.save_all_tracks()
        # Scene.plot_real_ghost_tracks(WithSemantic=WithSemantic, line=True)
        Scene.plot_all(WithSemantic=WithSemantic,plot_real=True,plot_ghost=plot_ghost,plot_vicon=plot_vicon, line=True,legend=False,set_axis=False,linewidth=2)


if __name__ == '__main__':
    mot(semantic_features=[], WithSemantic=True, withSRR=True)

