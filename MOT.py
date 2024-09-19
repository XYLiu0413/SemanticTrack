from TrackClasses.TracksManagement import *

from utils.common import load_json_scene_names, load_json_label_mapping, load_json_GroundTruth, \
    load_input_centroids_data_path, get_in_channels_and_extension, get_channels_and_extension

scene_names = load_json_scene_names()
label_mapping = load_json_label_mapping()
ground_truth_data = load_json_GroundTruth()


# def scene_process(scene_name, scene_data):
#     Scene = TracksModule(scene_name)
#     # data = loadmat('InputData/{}.mat'.format(scene_name))  # Load data specific to the scene
#     data_path=scene_data['input_LabelCentroids_pkl']
#     # Get files and parameters from the dictionary
#     Vicon_path = scene_data['file_path']
#     x_origin = scene_data['vicon_origin_x']
#     y_origin = scene_data['vicon_origin_y']
#     start_time = scene_data['start_timestamp']
#     end_time = scene_data['end_timestamp']
#     GTs = scene_data['GTs']
#     # Scene.data_process(data)
#     Scene.load_data(data_path)
#     Scene.initial_tracks(Scene.Centroids[0])
#
#     Scene.mot_process()
#     Scene.report()
#
#     # Define filenames based on scene
#
#     tracks_filename = 'OutputDataSC/TracksFile/{}_Tracks.pkl'.format(scene_name)
#     realtracks_filename = 'OutputDataSC/RealTracksFile/{}_RealTracks.pkl'.format(scene_name)
#     tracks_image_filename = 'OutputDataSC/ImagesAllTracks/{}.png'.format(scene_name)
#     realtracks_image_filename = 'OutputDataSC/ImagesRealTracks/{}.png'.format(scene_name)
#     real_gif_filename = 'OutputDataSC/RealTracks_gif/{}_RealTracks.gif'.format(scene_name)
#
#     Scene.write_Tracks(tracks_filename)
#     Scene.write_RealTracks(realtracks_filename)
#
#     # Scene.plot_combined_tracks1(Vicon_path, x_origin, y_origin, start_time, end_time, Scene.RealTracks)
#     Scene.plot_all_tracks(tracks_image_filename)
#     Scene.plot_realtracks(realtracks_image_filename)
#     Scene.animation(real_gif_filename)
#     # Scene.plot_P()
#     Scene.evaluation(Vicon_path, x_origin, y_origin, start_time, end_time,GTs)
#     return Scene.RealTracks,Scene.Centroids

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
    # for scene_name in scene_names[1:-1]:
    #     scene_data = ground_truth_data[scene_name]
    #     # Scenes=[]
    #     data_path = scene_data['input_Withlabel_Centroids_pkl']
    #     Vicon_path = scene_data['file_path']
    #     x_origin = scene_data['vicon_origin_x']
    #     y_origin = scene_data['vicon_origin_y']
    #     start_time = scene_data['start_timestamp']
    #     end_time = scene_data['end_timestamp']
    #     GTs = scene_data['GTs']
    #     Scene = TracksModule(name=scene_name)
    #     Scene.load_data(data_path)
    #     Scene.initial_tracks(Scene.Centroids[0])
    #     Scene.mot_process()
    #     # Scene.report(WithSemantic=True)
    #     Scene.report(WithSemantic=True)
    #     # Scene.plot_realtracks()
    #     # Scene.save_real_tracks()
    #     # Scene.save_ghost_tracks()
    #     # Scene.save_none_tracks()
    #     # Scene.save_all_tracks()
    #     Scene.plot_real_ghost_tracks(line=True)
    # Scenes.append(Scene)
