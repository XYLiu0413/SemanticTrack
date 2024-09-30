import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat
from TrackClasses.EKF import *  





# Initialize a track
class Track:
    """
    Used to record the objects tracked at each moment
    """
    def __init__(self,
                 track_id,
                 s0,
                 timestamp,
                 label,
                 u_id,
                 u,
                 max_skip_frames,  # maximum allowed frames to be skipped for the track object undetected
                 min_confirm_frames,  # min_confirm_length: minimum frames before new track is confirmed
                 G  
                 ):
        """

        :param track_id: identification of each track object starting from 1
        :param initial_point: The initial point of the trajectory
        :param timestamp:# 从0开始
        """
        self.track_id = track_id  # global track ID
        self.track_path = [s0]   # Initialize track path with the initial state
        self.track_path_fake=[s0]# Some tracks have several frames that are not detected in the middle, and this variable also contains the state that is not detected in the middle but we assume
        self.skipped_frames = 0  # number of frames skipped undetected
        self.confirm_frames = 1  # number of frames tracked=len（track_path）
        self.error_frames = 0
        self.labels=[label]
        self.association_uid= [u_id]  # Record the id of the centroid measurement associated with the track at that moment (frame)
        self.association_u=[u]     # Record the centroid measurement associated with the track at this moment (frame). If the corresponding id is 0, the measurement is created.
        self.track_state = 'DETECT'  # Initial state is 'detect
        self.start_time = timestamp  #The record track is the timestamp of the entire system (that is, the record time), because some targets may not be tracked until midway through the system, when the track is a FREE value
        self.end_time = timestamp+1
        self.max_skip_frames =max_skip_frames
        self.min_confirm_frames = min_confirm_frames
        self.G = G
        self.EKF=EKF(model_cv(),s0)

    def track_predict(self):
        self.EKF.ekf_predict()
        self.EKF.ekf_innovation()

    def track_update(self, u_associated):
        """
        :param u_associated: track_association()
                            The observation vector which is successfully associated and assigned to the track
        :return:
        """

        if self.track_state ==  'ACTIVE':
            if      self.skipped_frames<=self.max_skip_frames:
                    self.EKF.ekf_update(u_associated)
                    self.track_path_fake.append(self.EKF.s)
                    if  self.association_uid[-1] != -1:
                        self.track_path.append(self.EKF.s)
                    self.end_time+=1  #time+1
            elif    self.skipped_frames>self.max_skip_frames:
                    self.active2free()
                    self.end_time=self.end_time-self.max_skip_frames  #Update to the last timestamp that the track actually exists
                    for count in range(self.max_skip_frames+1):
                        if count<self.max_skip_frames:
                            self.track_path_fake.pop()  
                        self.association_u.pop()
                        self.association_uid.pop()
                        self.labels.pop()

        if self.track_state == 'DETECT':
            if      self.skipped_frames > 0:
                    self.detect2free()
            elif    self.skipped_frames == 0:
                    self.EKF.ekf_update(u_associated)
                    self.track_path_fake.append(self.EKF.s)
                    if self.association_uid[-1] != -1:
                        self.track_path.append(self.EKF.s)
                    if self.confirm_frames >= self.min_confirm_frames:
                        self.detect2active()
                    self.end_time += 1# self.end_time += 1


    def active2free(self):
        if self.track_state == 'ACTIVE':
            self.track_state = 'FREE'
    def detect2free(self):
        """Change track state from 'detect' to 'free'."""
        if self.track_state == 'DETECT':
            self.track_state = 'FREE'
        # else:
        #     print("The state of track",self.track_id," has been \"FREE\"")
    def detect2active(self):
        """Change track state from 'free' to 'detect'."""
        if self.track_state == 'DETECT':
            self.track_state = 'ACTIVE'
        # else:
        #     print("The state of track",self.track_id," has been \"DETECT\"")



def arraylist_comparision(list1,list2):
    """

    :param list1:
    :param list2:
    :return:True / False
    """
    if len(list1) != len(list2):
        return False

    for path1, path2 in zip(list1, list2):
        if not np.array_equal(path1, path2):
            return False

    return True

def CalMahalanobis2D(y, C):
    """
    Computes the 2-dimensional Mahalanobis distance^2 between vector y and distribution C.
    :param y: Vector y, a 1x3 NumPy array. [x,y,v] (only [x,y] will be used)
    :param C: Matrix C, a 3x3 NumPy matrix representing the inverse of error covariance matrix.
              (only the top-left 2x2 sub-matrix will be used)
    :return: md, the computed 2-dimensional Mahalanobis distance^2.
    """
    # Use only the first two elements of y and the 2x2 submatrix in the upper left corner of C
    y_2d = y[:2]
    C_2d = C[:2, :2]

    #Calculate the square of the two-dimensional Mahalanobis distance
    md = y_2d.T @ C_2d @ y_2d
    return md.item()  # Returns a scalar value rather than a set of elements
def CalMahalanobis(y, C):
    """
    Computes the 3-dimensional Mahalanobis distance between vector y and distribution C.
    :param y: Vector y, a 1x3 NumPy array. [x,y,v]
    :param C: Matrix C, a 3x3 NumPy matrix representing the inverse of error covariance matrix.
    :return: md(This is actually d^2), the computed 3-dimensional Mahalanobis distance^2.
    """
    # Calculate the Mahalanobis distance
    md = y.T @ C @ y
    return md.item()  # Returns a scalar value rather than a set of elements
def scoring(C, md):
    """
    :param C: The absolute value of the determinant of the matrix C
    :param md: Square of the Mahalanobis distance
    :return: Calculate the resulting score
    """
    CG=np.abs(np.linalg.det(C))
    return np.log(CG) + md



if __name__ == '__main__':

    # CGi_determinant = np.abs(np.linalg.det(np.array([[1, 2], [3, 4]])))  

    v = np.array([[1.0, 2.0, 3.0]])  # 1x3
    D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 3x3 
    md = CalMahalanobis(v, D)
    print("Mahalanobis distance:", md)

    #
    # def associate_tracks(tracks, radar_points):
    #     for point in radar_points:
    #         distances = [Track.distance(point, track.get_latest_point()) for track in tracks]
    #         closest_track_idx = np.argmin(distances) if distances else None
    #         if closest_track_idx is not None and distances[closest_track_idx] < ASSOCIATION_THRESHOLD:
    #             tracks[closest_track_idx].ekf_update(point)
    #         else:
    #             tracks.append(Track(len(tracks), point))
    #
    #     for track in tracks:
    #         track.increment_no_update()
    #
    # def track_targets(radar_frames):
    #     tracks = []
    #     for frame in radar_frames:
    #         associate_tracks(tracks, frame)
    #         tracks = [track for track in tracks if track.is_active]
    #     return tracks
    #
    # # Constants
    # MAX_MISSED_UPDATES = 5
    # ASSOCIATION_THRESHOLD = 10.0  # Threshold for associating points to the same track
    #
    # # Example usage
    # radar_data_frames = [
    #     [[1, 1, 2], [5, 5, 1]],
    #     [[1.1, 1.1, 2], [5.1, 5, 1]],
    #     # ... more frames
    # ]
    #
    # tracked_targets = track_targets(radar_data_frames)
    # for track in tracked_targets:
    #     print(f"Track {track.Track_id}: {track.points}")
