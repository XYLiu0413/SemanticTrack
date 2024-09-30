import json
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt

from Evaluation.evaluation_common import vicon_data, gt_data
from TrackClasses.TrackManagement import *
from TrackClasses.common import x_dot, has_different_frequencies, get_label_id
with open('GroundTruth/label_mapping.json', 'rb') as f:
    label_mapping = json.load(f)


class TracksModule():
    def __init__(self,
                 name,
                 extension,
                 Tracks=None,
                 RealTracks=None,
                 GhostTracks=None,
                 NoneTracks=None,
                 valid_tracks_indices=None,
                 ghost_tracks_indices=None,
                 none_tracks_indices=None,
                 NotFreeTracks=None,
                 Centroids=None,
                 Track_id=0,
                 Timestamp=0,
                 Initial_vy=1,
                 Gate=np.linalg.norm([1, 1, 1]),
                 Min_target_frames=15,
                 Max_skip_frames=3,
                 Min_confirm_frames=5,
                 Max_error_frames=3,
                 semantic_features=None

                 ):
        """
        Each forecast and updated track is NotFreeTracks
        :param name:  The name of the tracing scene
        :param Tracks: All created tracks
        :param RealTracks:track for real targets
        :param GhostTracks:track the ghost targets
        :param NotFreeTracks:tracks that record the status of each round when it needs to be updated are not 'FREE'
        :param Centroids:Input centroid data
        :param Track_id:The unique id of track
        :param Timestamp:
        :param Initial_vy:Positive values mean away from radar, negative values mean close to radar. The state quantity S0 at the initial track, and an initial vy needs to be set to calculate vx information
        :param Gate: 
        :param Min_target_frames:The minimum number of frames confirmed as the true target track
        :param Max_skip_frames:  maximum allowed frames to be skipped for the track object undetected
        :param Min_confirm_frames:  min_confirm_length: minimum frames before new track is confirmed
        :param Max_error_frames: 
        """
        if Centroids is None:
            Centroids = []
        if NotFreeTracks is None:
            NotFreeTracks = []
        if GhostTracks is None:
            GhostTracks = []
        if NoneTracks is None:
            NoneTracks = []
        if ghost_tracks_indices is None:
            ghost_tracks_indices = []
        if none_tracks_indices is None:
            none_tracks_indices = []
        if valid_tracks_indices is None:
            valid_tracks_indices = []
        if RealTracks is None:
            RealTracks = []
        if Tracks is None:
            Tracks = []
        if semantic_features is None:
            semantic_features = ['v','RCS']
        self.name = name
        self.extension = extension
        self.Tracks = Tracks
        self.RealTracks = RealTracks
        self.GhostTracks = GhostTracks
        self.NoneTracks = NoneTracks
        self.valid_tracks_indices = valid_tracks_indices
        self.none_tracks_indices = none_tracks_indices
        self.ghost_tracks_indices = ghost_tracks_indices
        self.NotFreeTracks = NotFreeTracks
        self.Centroids = Centroids
        self.Track_id = Track_id
        self.Timestamp = Timestamp
        self.Initial_vy = Initial_vy
        self.Gate = Gate
        self.Min_target_frames = Min_target_frames
        self.Max_skip_frames = Max_skip_frames
        self.Min_confirm_frames = Min_confirm_frames
        self.Max_error_frames = Max_error_frames
        self.semantic_features = semantic_features

    def load_data(self, CentroidsPath):
        with open(CentroidsPath, 'rb') as f:
            self.Centroids = pickle.load(f)

    def allocation(self, centroid):
        if np.size(centroid) != 0:
            for u in centroid:
                if u[5] == 0:
                    vx = x_dot(u[0], u[1], u[3], self.Initial_vy)  # vy(=y_dot)The convention is initialized with 1
                    self.Tracks.append(Track(
                        track_id=self.Track_id,
                        s0=np.array([u[0], u[1], vx, self.Initial_vy]),
                        label=u[6],
                        u_id=u[4],
                        u=u,
                        timestamp=self.Timestamp,
                        max_skip_frames=self.Max_skip_frames,
                        min_confirm_frames=self.Min_confirm_frames,
                        G=self.Gate
                    )
                    )
                    self.Track_id += 1

    def initial_tracks(self, centroid):
        if np.size(centroid) != 0:
            self.allocation(centroid)
            self.update_tracks()
            self.time_run()

    def update_tracks(self):
        self.NotFreeTracks = [t for t in self.Tracks if t.track_state != 'FREE']

    def mot_process(self):
        for centroid in self.Centroids[1:]:
            self.module_predict()
            self.hungarian_association(centroid)
            self.module_update()
            self.allocation(centroid)
            self.update_tracks()
            self.time_run()

    def module_predict(self):
        for track in self.NotFreeTracks:
            # if track.track_state is not 'FREE':
            track.track_predict()

    def hungarian_association(self, centroid):
        num_tracks = len(self.NotFreeTracks)
        num_u = len(centroid)
        if np.size(centroid) != 0:
            cost_matrix = np.zeros((num_tracks, num_u))
            INF = 1e9
            # Construct the cost matrix and traverse the tracks in self.Tracks that are not in a 'FREE' state
            for i, track in enumerate(self.NotFreeTracks):
                for j, u in enumerate(centroid):
                    distance = np.linalg.norm(track.EKF.u_apr - np.array(u[[0, 1, 3]]))
                    cost_matrix[i, j] = distance if distance <= self.Gate else INF

            # Find the best match using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)


            associated_centroids = set()  
            for i, j in zip(row_ind, col_ind):  
                if cost_matrix[i, j] != INF:
                    u = centroid[j]
                    u[5] = self.NotFreeTracks[i].track_id
                    self.NotFreeTracks[i].association_uid.append(u[4])  
                    self.NotFreeTracks[i].association_u.append(u)
                    self.NotFreeTracks[i].labels.append(u[6])
                    self.NotFreeTracks[i].confirm_frames += 1
                    self.NotFreeTracks[i].skipped_frames = 0
                    associated_centroids.add(j) 
                else:
                   
                    self.NotFreeTracks[i].association_uid.append(-1)
                    self.NotFreeTracks[i].association_u.append(self.NotFreeTracks[i].EKF.u_apr)
                    self.NotFreeTracks[i].labels.append(-1)
                    self.NotFreeTracks[i].skipped_frames += 1
                    self.NotFreeTracks[i].confirm_frames = 0
            # Processing unsuccessful associated tracks
            for i in range(num_tracks):
                if i not in row_ind:
                    self.NotFreeTracks[i].association_uid.append(-1)
                    self.NotFreeTracks[i].association_u.append(self.NotFreeTracks[i].EKF.u_apr)
                    self.NotFreeTracks[i].labels.append(-1)
                    self.NotFreeTracks[i].skipped_frames += 1
                    self.NotFreeTracks[i].confirm_frames = 0
            # Index of the unassociated centroids
            unassociated_centroids = set(range(num_u)) - associated_centroids
            return list(associated_centroids), list(unassociated_centroids)
        # This frame does not have a centroid, and is counted as having no correlation success
        else:
            for i in range(len(self.NotFreeTracks)):
                self.NotFreeTracks[i].association_uid.append(-1)
                self.NotFreeTracks[i].association_u.append(self.NotFreeTracks[i].EKF.u_apr)
                self.NotFreeTracks[i].labels.append(-1)
                self.NotFreeTracks[i].skipped_frames += 1
                self.NotFreeTracks[i].confirm_frames = 0



    def module_update(self):
        for track in self.NotFreeTracks:
            u = track.association_u[-1]
            if np.size(u) == 3:
                track.track_update(u)
            else:
                track.track_update(u[[0, 1, 3]])

    def time_run(self):
        self.Timestamp += 1

    def report(self, WithSemantic=True):
        # valid_tracks_indices=[]
        # ghost_tracks_indices=[]
        if not WithSemantic:
            for i, track in enumerate(self.Tracks):
                # print(len(track.track_path))
                if len(track.track_path) < self.Min_confirm_frames:
                    track.track_state = 'FREE'
                    self.none_tracks_indices.append(i)
                    self.NoneTracks.append(track)
                    # print("track",track.track_id," is ghost target")

                elif len(track.track_path) < self.Min_target_frames:
                    track.track_state = 'FREE'
                    self.ghost_tracks_indices.append(i)
                    self.GhostTracks.append(track)

                else:
                    self.valid_tracks_indices.append(i)
                    self.RealTracks.append(track)
                    print("\ntrack", track.track_id, " is real target")
                    print("its state is", track.track_state, 'and length is', len(track.track_path))
                    print("start time ", track.start_time, 'to  end time ', track.end_time)
                    print('The label of each measurement is', track.labels)
                    # print("The id of the per-frame measurement associated with it is ", track.association_uid)

            print("\n A total of", len(self.Tracks), "tracks")
            print("A total of", len(self.valid_tracks_indices), "real target tracks")
            print("A total of", len(self.ghost_tracks_indices), "ghost target tracks(False targets)")
            print("A total of", len(self.none_tracks_indices), "none target tracks")
        if WithSemantic:
            for i, track in enumerate(self.Tracks):
                if has_different_frequencies(track.labels):
                    label_id = get_label_id(track.labels)
                    if label_id == 2 or label_id == -1:
                        track.track_state = 'FREE'
                        self.ghost_tracks_indices.append(i)
                        self.GhostTracks.append(track)
                    elif (label_id == 0 or label_id == 1) and len(track.track_path) >= self.Min_target_frames:
                        self.valid_tracks_indices.append(i)
                        self.RealTracks.append(track)
                        print("\ntrack", track.track_id, " is real target")
                        print("its state is", track.track_state, 'and length is', len(track.track_path))
                        print("start time ", track.start_time, 'to  end time ', track.end_time)
                        print('The label of each measurement is', track.labels)
                    else:
                        self.none_tracks_indices.append(i)
                        self.NoneTracks.append(track)
                else:
                    self.none_tracks_indices.append(i)
                    self.NoneTracks.append(track)
            print("\n A total of", len(self.Tracks), "tracks")
            print("A total of", len(self.valid_tracks_indices), "real target tracks")
            print("A total of", len(self.none_tracks_indices), "none target tracks")
            print("A total of", len(self.ghost_tracks_indices), "Ghost target tracks" )


    # def report_(self):
    def plot_realtracks(self):
        """
        plot RealTracks
        :return:
        """
        if self.RealTracks:


            fig, ax = plt.subplots()

            colors = plt.cm.jet(np.linspace(0, 1, len(self.RealTracks)))
            label_counts = {}
            for i, track in enumerate(self.RealTracks):

                x, y, _, _ = np.array(track.track_path_fake).T
                counter = Counter(track.labels)
                most_common = counter.most_common(1)
                label_id = most_common[0][0]
                for key, value in label_mapping.items():
                    if value == label_id:
                        base_label = key
                
                if base_label not in label_counts:
                    label_counts[base_label] = 0
                label_counts[base_label] += 1
                label = f"{base_label}{label_counts[base_label]}"
                x_new = x[track.labels == label_id]
                y_new = y[track.labels == label_id]
                ax.scatter(x_new, y_new, color=colors[i], label=label)
            ax.legend()
            # ax.get_legend().remove()
            ax.set_xlim(-10, 10)
            ax.set_ylim(0, 15)
            plt.title('RealTracks Paths')
            plt.xlabel('X/m')
            plt.ylabel('Y/m')
            plt.show()

    def plot_real_ghost_tracks(self, WithSemantic=True, line=False):

        global base_label
        fig, ax = plt.subplots()
        if self.NoneTracks:
            for i, track in enumerate(self.NoneTracks):


                x, y, _, _ = np.array(track.track_path_fake).T
                if not line:
                    ax.scatter(x, y, color='gray')
                else:
                    ax.plot(x, y, color='gray', linestyle='--', marker='', markersize=2)
        if self.GhostTracks:
            # colors = plt.cm.jet(np.linspace(0, 1, len(self.RealTracks)))
            # label_counts = {}
            for i, track in enumerate(self.GhostTracks):

                x, y, _, _ = np.array(track.track_path_fake).T
                if not line:
                    ax.scatter(x, y, color='gray')
                else:
                    ax.plot(x, y, color='gray', linestyle='--', marker='o', markersize=2,linewidth=1)

        if self.RealTracks:

            colors = plt.cm.jet(np.linspace(0, 1, len(self.RealTracks)))
            label_counts = {}
            for i, track in enumerate(self.RealTracks):


                x, y, _, _ = np.array(track.track_path_fake).T
                x_new = x
                y_new = y
                label = f'RealTrack{i + 1}'
                if WithSemantic:
                    counter = Counter(track.labels)
                    most_common = counter.most_common(1)
                    label_id = most_common[0][0]
                    for key, value in label_mapping.items():
                        if value == label_id:
                            base_label = key

                    if base_label not in label_counts:
                        label_counts[base_label] = 0
                    label_counts[base_label] += 1
                    label = f"{base_label}{label_counts[base_label]}"
                    x_new = x[track.labels == label_id]
                    y_new = y[track.labels == label_id]
                if not line:
                    ax.scatter(x_new, y_new, color=colors[i], label=label)
                else:
                    ax.plot(x_new, y_new, color=colors[i], linestyle='-', marker='o', markersize=1, label=label)
            ax.legend()
            # ax.get_legend().remove()
            ax.set_xlim(-10, 10)
            ax.set_ylim(0, 15)
            plt.title('RealTracks Paths')
            plt.xlabel('X/m')
            plt.ylabel('Y/m')
            plt.show()

    def plot_all(self, WithSemantic=True, line=True, plot_real=True, plot_ghost=True, plot_vicon=True, legend=False,
                 set_axis=False,linewidth=1):
        fig, ax = plt.subplots(dpi=300)
        if self.extension!='':
            origin_scene_name=self.name[:-len(self.extension)]
        else:
            origin_scene_name=self.name
        if plot_ghost:
            if self.NoneTracks:
                for i, track in enumerate(self.NoneTracks):
                    if len(track.track_path) > 10:
                        continue

                    x, y, _, _ = np.array(track.track_path_fake).T
                    if not line:
                        ax.scatter(x, y, color='gray',s=2)
                    else:
                        ax.plot(x, y, color='gray', linestyle='--', marker='', markersize=5,linewidth=linewidth)
            if self.GhostTracks:
                # colors = plt.cm.jet(np.linspace(0, 1, len(self.RealTracks)))
                # label_counts = {}
                for i, track in enumerate(self.GhostTracks):
                    # if len(track.track_path)>10:
                    #     continue

                    x, y, _, _ = np.array(track.track_path_fake).T
                    if not line:
                        ax.scatter(x, y, color='gray')
                    else:
                        ax.plot(x, y, color='gray', linestyle='--', marker='', markersize=2, linewidth=linewidth)

        if plot_vicon:
            # gt_x, gt_y = vicon_data(self.name)
            gt_x, gt_y = gt_data(origin_scene_name)
            colors = plt.cm.jet(np.linspace(0, 1, len(gt_x)))
            if origin_scene_name == 'Multiperson_vehicle':
                colors[[0,1]] = colors[[1,0]]
            for i, (x, y) in enumerate(zip(gt_x.values(), gt_y.values())):
                ax.plot(x, y, color=colors[i], linestyle=':', marker='', markersize=1,linewidth=linewidth)
        if plot_real:
            if self.RealTracks:
                colors = plt.cm.jet(np.linspace(0, 1, len(self.RealTracks)))
                if origin_scene_name == 'Multiperson_vehicle':
                    colors[[0,1]] = colors[[1,0]]
                label_counts = {}
                for i, track in enumerate(self.RealTracks):
                    if origin_scene_name == 'Single_person' and i==1:
                        continue

                    x, y, _, _ = np.array(track.track_path_fake).T
                    x_new = x
                    y_new = y
                    label = f'RealTrack{i + 1}'
                    if WithSemantic:
                        counter = Counter(track.labels)
                        most_common = counter.most_common(1)
                        label_id = most_common[0][0]
                        for key, value in label_mapping.items():
                            if value == label_id:
                                base_label = key
    
                        if base_label not in label_counts:
                            label_counts[base_label] = 0
                        label_counts[base_label] += 1
                        label = f"{base_label}{label_counts[base_label]}"
                        x_new = x[track.labels == label_id]
                        y_new = y[track.labels == label_id]
                    if not line:
                        ax.scatter(x_new, y_new, color=colors[i], label=label)
                    else:
                        ax.plot(x_new, y_new, color=colors[i], linestyle='-', marker='', markersize=1,linewidth=linewidth, label=label)

        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 15)

        ax.set_xticks(range(-10, 11, 5))  
        ax.set_yticks(range(0, 16, 5))  
        ax.grid(True)

        ax.spines['top'].set_color('black')
        ax.spines['top'].set_linewidth(5)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(5)
        ax.spines['right'].set_color('black')
        ax.spines['right'].set_linewidth(5)

        ax.set_xticklabels([])
        ax.set_yticklabels([])


        if legend:
            ax.legend()
        if set_axis:
            ax.axis('off')
            # plt.title('RealTracks Paths')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.show()

    def save_real_tracks(self):
        """
        保存RealTracks到文件
        :param filename: 
        """
        base_path = f'OutputData/RealTracks/{self.name}'
        file_extension = '.pkl'
        file_path = f'{base_path}{file_extension}'

        i = 1
        while os.path.exists(file_path):
            file_path = f'{base_path}_{i}{file_extension}'
            i += 1
        with open(file_path, 'wb') as f:
            pickle.dump(self.RealTracks, f)
        print(f"RealTracks saved to {file_path}")

    def save_ghost_tracks(self):
        """
        
        :param filename: 
        """
        base_path = f'OutputData/GhostTracks/{self.name}'
        file_extension = '.pkl'
        file_path = f'{base_path}{file_extension}'
        
        i = 1
        while os.path.exists(file_path):
            file_path = f'{base_path}_{i}{file_extension}'
            i += 1
        with open(file_path, 'wb') as f:
            pickle.dump(self.GhostTracks, f)
        print(f"RealTracks saved to {file_path}")

    def save_none_tracks(self):
        base_path = f'OutputData/NoneTracks/{self.name}'
        file_extension = '.pkl'
        file_path = f'{base_path}{file_extension}'
       
        i = 1
        while os.path.exists(file_path):
            file_path = f'{base_path}_{i}{file_extension}'
            i += 1
        with open(file_path, 'wb') as f:
            pickle.dump(self.NoneTracks, f)
        print(f"RealTracks saved to {file_path}")

    def save_all_tracks(self):
        base_path = f'OutputData/Tracks/{self.name}'
        file_extension = '.pkl'
        file_path = f'{base_path}{file_extension}'
        
        i = 1
        while os.path.exists(file_path):
            file_path = f'{base_path}_{i}{file_extension}'
            i += 1
        with open(file_path, 'wb') as f:
            pickle.dump(self.Tracks, f)
        print(f"RealTracks saved to {file_path}")


if __name__ == "__main__":
    Track(1, np.array([1, 2, 3, 4]), 0, 0)
