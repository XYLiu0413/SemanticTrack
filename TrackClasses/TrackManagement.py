import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat
from TrackClasses.EKF import *  # 直接导入模块“所有“变量和函数
# 本质读取模块的__all__属性，看这个属性里定义了哪些变量和函数
# 如果模块里没有定义__all__属性，则读取所有不以一个下划线(_)开始的变量和函数
#__all__=['Track']
# # Gate Parameters
# max_skip_frames = 3 # maximum allowed frames to be skipped for the track object undetected
# min_confirm_frames = 5 #min_confirm_length: minimum frames before new track is confirmed
# G=1.5*1.5*4                  #定义的最小马氏距离门限
# # For example, for people counting, it is expected that the limits are set based on human body dimensions and dynamicity limits: (ex,
# # 1.5x1.5x2 m in x * y, and 4m/s of Doppler spread).




# Initialize a track
class Track:
    """成飞
    用于记录每个时刻跟踪的对象
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
                 G  # 定义的最小马氏距离门限
                 ):
        """

        :param track_id: identification of each track object starting from 1
        :param initial_point: The initial point of the trajectory
        :param timestamp:# 从0开始
        """
        self.track_id = track_id  # global track ID
        self.track_path = [s0]   # 真实的记录属于track的每一帧的状态Initialize track path with the initial state
        self.track_path_fake=[s0]# 有的track中间有没有探测到的几帧，该变量也包含中间未探测但是我们假设出来的状态
        self.skipped_frames = 0  # number of frames skipped undetected
        self.confirm_frames = 1  # number of frames tracked确定属于该track的帧数，=len（track_path）
        self.error_frames = 0
        self.labels=[label]
        self.association_uid= [u_id]  # 记录该时刻（该帧）track关联的质心测量量的id
        self.association_u=[u]     # 记录该时刻（该帧）track关联的质心测量量,若对应为id为0，则该测量量为创造出来的。
        #self.score = float('inf')# track在该时刻（该帧）与关联的测量量的最优打分值
        self.track_state = 'DETECT'  # Initial state is 'detect
        #根据timestamp和association_uid可以知道该track
        self.start_time = timestamp  #记录track处于整个系统的时间戳（即记录时间），因为有的目标可能从系统中途才开始被跟踪，当track为FREE数值
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
        :param u_associated: track_association()的返回值做输入
                            The observation vector which is successfully associated and assigned to the track
        :return:
        """

        if self.track_state ==  'ACTIVE':
            if      self.skipped_frames<=self.max_skip_frames:
                    self.EKF.ekf_update(u_associated)
                    self.track_path_fake.append(self.EKF.s)
                    if  self.association_uid[-1] != -1:
                        self.track_path.append(self.EKF.s)
                    self.end_time+=1  #执行完一个周期时间戳+1
            elif    self.skipped_frames>self.max_skip_frames:
                    self.active2free()
                    self.end_time=self.end_time-self.max_skip_frames  #更新至track真实存在的最后的timestamp
                    for count in range(self.max_skip_frames+1):
                        if count<self.max_skip_frames:
                            self.track_path_fake.pop()  #少弹出一次，以保证track_path_fake长度与labels一致
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

    :param list1: 列表1元素均为数组类型
    :param list2: 列表2元素均为数组类型
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
    # 只使用y的前两个元素和C的左上角的2x2子矩阵
    y_2d = y[:2]
    C_2d = C[:2, :2]

    # 计算二维马氏距离的平方
    md = y_2d.T @ C_2d @ y_2d
    return md.item()  # 返回一个标量值而不是单元素数组
def CalMahalanobis(y, C):
    """
    Computes the 3-dimensional Mahalanobis distance between vector y and distribution C.
    :param y: Vector y, a 1x3 NumPy array. [x,y,v]
    :param C: Matrix C, a 3x3 NumPy matrix representing the inverse of error covariance matrix.
    :return: md(这里其实计算出来的是d^2), the computed 3-dimensional Mahalanobis distance^2.
    """
    # 计算马氏距离
    md = y.T @ C @ y
    return md.item()  # 返回一个标量值而不是单元素数组
def scoring(C, md):
    """
    :param C: 矩阵C的行列式的绝对值
    :param md: 马氏距离的平方
    :return: 计算得到的得分
    """
    CG=np.abs(np.linalg.det(C))
    return np.log(CG) + md



if __name__ == '__main__':

    # 示例输入
    # CGi_determinant = np.abs(np.linalg.det(np.array([[1, 2], [3, 4]])))  # 这里使用了一个2x2矩阵的行列式作为例子

    #示例用法
    v = np.array([[1.0, 2.0, 3.0]])  # 1x3 NumPy矩阵
    D = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 3x3 NumPy矩阵
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
