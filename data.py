#!/usr/bin/env python3

import rospy
import tf2_ros
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, inf, atan2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from visualizer import RvizMarker
from features import FeatureDetection
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class EkfFilter:
    def __init__(self) -> None:
        self.feature_detection = FeatureDetection()
        self.visualizer = RvizMarker()
        self.rate = rospy.Rate(30)
        self.landmarks = []
        self.potencial_landmarks = []
        self.low_dim_jacobs = []
        self.Rt = np.array([[0.00001, 0, 0], [0, 0.00001, 0], [0, 0, 0.0000005]])
        self.Q = np.array([[0.01, 0], [0, 0.1]])
        self.association_threshold = 0.01  
        self.landmarks_initialized = False
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.listened = False

    def broadcast_map(self):
        for object in self.landmarks:
            self.visualizer.publish_line(object)
        
    def predict_step(self, state, covariance, odometry):
        n = len(state)

        #[dt, vx, vy, v_yaw] = odometry
        [delta_rot_1, delta_trans, delta_rot_2] = odometry
        angle = state[2]
        
        #B = np.array([[cos(angle), -1*sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]], dtype=np.float64)
        #u = np.array([vx, vy, v_yaw])
        B = np.array([delta_trans*np.cos(angle + delta_rot_1), delta_trans * np.sin(angle + delta_rot_1), delta_rot_1 + delta_rot_2])
        #new_state = state[:3]
        #state[:3] = state[:3] + dt * B.dot(u) #Update previous state
        #state[:3] = new_state + B
        state[:3] = state[:3] + B
        #G_x = np.array([[1, 0, -1*(vx* sin(angle) + vy * cos(angle))*dt],  #Jacobian
        #              [0, 1, (vx * cos(angle) - vy * sin(angle))*dt],
        #              [0, 0, 1]],  dtype=np.float64)

        G_x = np.array([[1, 0, -1*delta_trans*np.sin(angle + delta_rot_1)],
                        [0, 1, delta_trans*np.cos(angle + delta_rot_1)],
                        [0, 0, 1]])

        pos_covariance = covariance[:3, :3]
        pos_landmark_covariance = covariance[:3, 3: n]

        pos_covariance[:] = G_x.dot(pos_covariance.dot(G_x.T)) + self.Rt #Update Covariance matrixes 
        pos_landmark_covariance[:] = G_x.dot(pos_landmark_covariance)
        covariance[3:n, :3] = pos_landmark_covariance.T

        return state, covariance

    def get_tf_to_map(self, time_stamp):
        while True:
            try:
                if self.listened == False:
                    trans = self.tf_buffer.lookup_transform("odom", "base_scan", time_stamp, rospy.Duration(0.1))   # slam -> base_ekf -> base_ekf_scan
                    self.listened = True
                else:
                    trans = self.tf_buffer.lookup_transform("odom", "base_efk_scan", time_stamp, rospy.Duration(0.1))
                _ ,_ , angle = euler_from_quaternion([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
                pose = np.array([trans.transform.translation.x, trans.transform.translation.y, angle])
                return pose
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException):
                print("STUCK")
                continue
    
    def transform_points(self, transform, points):
        points_array = np.array([[], []])
        for point in points:
            col = np.array([[point[0]], [point[1]]])
            points_array = np.append(points_array, col, axis=1)
        
        angle = transform[2]
        rotation = np.array([[cos(angle), -1*sin(angle)], [sin(angle), cos(angle)]])
        points_array = np.add(rotation.dot(points_array), np.array([[transform[0]], [transform[1]]])).T
        return points_array
    
    def get_predicted_measurement(self, landmark, laser_position):   # pr, ptheta
        angle = landmark[1]
        M = np.array([[-1*cos(angle), -1*sin(angle), 0],
                          [0, 0, -1]], dtype=np.float64)
        
        predicted_measurement = landmark + M.dot(laser_position)    ## [5 pi] - 
        return predicted_measurement
    
    def calculate_low_dim_jacobian(self, landmark, laser_position):
        cos_p_theta = np.cos(landmark[1])
        sin_p_theta = np.sin(landmark[1])
        sin_l_theta = np.sin(laser_position[2])
        cos_l_theta = np.cos(laser_position[2])
        #return np.array([[-1*cos_p_theta, -1*sin_p_theta, (cos_p_theta * sin_l_theta - sin_p_theta * cos_l_theta)*0.064, 1, sin_p_theta * (laser_position[0] + 0.064 + cos_l_theta * 0.064) - cos_p_theta * (laser_position[1] + sin_l_theta * 0.064)], 
        #                            [0, 0, -1, 0, 1]], dtype=np.float64)
        return np.array([[-1*cos_p_theta, -1*sin_p_theta, (cos_p_theta * sin_l_theta - sin_p_theta * cos_l_theta)*0.064, 1, sin_p_theta * (laser_position[0] + cos_l_theta * 0.064) - cos_p_theta * (laser_position[1] + sin_l_theta * 0.064)], 
                                    [0, 0, -1, 0, 1]], dtype=np.float64)
    
    def create_expansion_matrix(self, state_length, landmark_index):
        expansion_matrix = np.zeros((5, state_length))
        expansion_matrix[:3,:3] = np.eye(3)
        expansion_matrix[3, 2*landmark_index + 3] = 1
        expansion_matrix[4, 2*landmark_index + 4] = 1
        return expansion_matrix
    
    def upgrade_feature(self, potencial_landmark, current_stamp):
        landmark_index = 0
        for potencial in self.potencial_landmarks:
            if (current_stamp - potencial[2]).to_sec() >= 10:
                self.potencial_landmarks.pop(landmark_index)
                landmark_index -= 1
            landmark_index += 1
            
        for landmark_index in range(0, len(self.potencial_landmarks)):
            if self.feature_detection.dist2pol(potencial_landmark, self.potencial_landmarks[landmark_index][0]) <= 0.03:
                if self.potencial_landmarks[landmark_index][1] == 2:
                    self.potencial_landmarks.pop(landmark_index)
                    return True
                else:
                    self.potencial_landmarks[landmark_index][1] += 1
                    return False
        
        self.potencial_landmarks.append([potencial_landmark, 1, current_stamp])
        return False
    
    def initialize_new_landmark(self, feature, transformation, predicted_state, predicted_covariance, stamp):
        previous_state_length = len(predicted_state)
        line_points_map = self.transform_points(transformation, feature[1])
        line_eq_map = self.feature_detection.points_2line(line_points_map[0], line_points_map[1])
        origin_projected = self.feature_detection.projection_point2line([0,0], line_eq_map[0], line_eq_map[1])
        potencial_landmark = self.feature_detection.cart2pol(origin_projected)

        if self.upgrade_feature(potencial_landmark, stamp):

            #self.visualizer.publish_line(line_points_map)
            #self.visualizer.publish_point(origin_projected)
            #self.visualizer.publish_laser_points(self.transform_points(transformation, self.feature_detection.laser_points))
            
            new_covariance = np.zeros((previous_state_length + 2, previous_state_length + 2))
            new_covariance[:previous_state_length, :previous_state_length] = predicted_covariance
            new_covariance[previous_state_length, previous_state_length] = 0.01                   
            new_covariance[previous_state_length + 1, previous_state_length + 1] = 0.01              
            predicted_state = np.append(predicted_state, self.feature_detection.cart2pol(origin_projected))
            self.landmarks.append(line_points_map)
            return predicted_state, new_covariance, 1

        return predicted_state, predicted_covariance, 0
    
    def calculate_matching_matrixes(self, laser_position, landmark_state, predicted_covariance, state_length, landmark_index):
        
        landmark_low_dim_jacobian = self.calculate_low_dim_jacobian(landmark_state, laser_position)
        expansion_matrix = self.create_expansion_matrix(state_length, landmark_index)
        high_dim_jacobian = landmark_low_dim_jacobian.dot(expansion_matrix)
        S = high_dim_jacobian.dot(predicted_covariance).dot(high_dim_jacobian.T) + self.Q
        S_inverse = np.linalg.inv(S)

        return S_inverse, high_dim_jacobian
    
    def find_association(self, feature, laser_position, predicted_state, predicted_covariance, stamp):
        state_length = len(predicted_state)
        minimum = inf
        for landmark_index in range(0, len(self.landmarks)):
            landmark_state = predicted_state[2*landmark_index +3: 2*landmark_index +5]
            #print("LANDMARK STATE: " + str(landmark_state))
            predicted_measurement = self.get_predicted_measurement(landmark_state, laser_position)
            if predicted_measurement[0] < 0:
                continue
            innovation = (feature[0] - predicted_measurement).reshape(2,1)
            diff = abs(feature[0][1] - predicted_measurement[1])
            clock_diff = 2*np.pi - diff
            innovation[1][0] = min(diff, clock_diff)
            innovation[1][0] = -1*innovation[1][0] if feature[0][1] < predicted_measurement[1] else innovation[1][0]
            S_inverse, high_dim_jacobian = self.calculate_matching_matrixes(laser_position, landmark_state, predicted_covariance, state_length, landmark_index)
            #if abs(innovation.T.dot(S_inverse).dot(innovation)[0]) <= self.association_threshold:
            distance = self.feature_detection.dist2pol(feature[0], predicted_measurement)
            if self.feature_detection.dist2pol(feature[0], predicted_measurement) <= minimum:
                minimum = distance
                index = landmark_index
                keeper = [predicted_state, predicted_covariance, innovation.reshape(2,), S_inverse, high_dim_jacobian, 1]
                

        if minimum <= 0.25:
            self.visualizer.publish_point(self.transform_points(laser_position, [self.feature_detection.pol2cart(feature[0])])[0], 0, 1, 0)
            self.landmarks[index] = np.append(self.landmarks[index], self.transform_points(laser_position, feature[1]), axis = 0)
            return keeper
        
        predicted_state, predicted_covariance, flag = self.initialize_new_landmark(feature, laser_position, predicted_state, predicted_covariance, stamp)
        if flag:
            S_inverse, high_dim_jacobian = self.calculate_matching_matrixes(laser_position, predicted_state[state_length:state_length+2],predicted_covariance, state_length + 2, len(self.landmarks) - 1)
            return predicted_state, predicted_covariance, np.array([0,0]), S_inverse, high_dim_jacobian, 1
        else:
            return predicted_state, predicted_covariance, 0, 0, 0, 0


    def normalize_state_vector(self, predicted_state):
        for landmark_index in range(0, len(self.landmarks)):
            landmark_state = predicted_state[2*landmark_index +3: 2*landmark_index +5]
            landmark_state[0] = abs(landmark_state[0])
            landmark_state[1] = landmark_state[1] % (2*np.pi)
        return predicted_state
    
    def correction_step(self, predicted_state, predicted_covariance, laser_scan, stamp):
        features = self.feature_detection.feature_extraction(laser_scan)
        laser_position = self.get_tf_to_map(stamp)
        #laser_position = predicted_state[:3] #+ np.array([-0.064, 0, 0])
        self.visualizer.publish_laser_points(self.transform_points(laser_position, self.feature_detection.laser_points)) #Visaulizador de pontos no referencial do mapa

        for feature in features:
            predicted_state, predicted_covariance, innovation, S_inverse, high_dim_jacobian, flag = self.find_association(feature, laser_position, predicted_state, predicted_covariance, stamp)
            if flag:
                kalman_gain = predicted_covariance.dot(high_dim_jacobian.T).dot(S_inverse)
                predicted_state = predicted_state + kalman_gain.dot(innovation)
                predicted_covariance = (np.eye(len(predicted_covariance)) - kalman_gain.dot(high_dim_jacobian)).dot(predicted_covariance)
                
                predicted_state = self.normalize_state_vector(predicted_state)
        
        return predicted_state, predicted_covariance

class Sub:
    def __init__(self, topic_name_odometry, topic_name_laser):
        self.visualizer = RvizMarker()
        self.feature_detection = FeatureDetection()
        self.points_plot = []  
        self.filter = EkfFilter()
        self.covariance_matrix = np.array([[0,0,0], [0,0,0], [0,0,0]], dtype=np.float64)
        self.state = np.array([0, 0, 0], dtype=np.float64)
        self.correction_covariance_matrix = self.covariance_matrix
        self.correction_state = self.state
        self.extracted = False
        self.moved = False
        self.laser_data = False
        self.correcting = False
        self.laser_scan = []
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(30)
        rospy.Subscriber(topic_name_odometry, Odometry, self.odom, queue_size=1000)
        rospy.Subscriber(topic_name_laser, LaserScan, self.laser, queue_size=1000)

    def broadcast_tf(self, frame_id, child_frame_id, timestamp, pose):
        t = TransformStamped()
        
        t.header.stamp = timestamp
        t.header.frame_id = frame_id
        
        t.child_frame_id = child_frame_id

        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = 0.0
        
        q = quaternion_from_euler(0, 0, pose[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def laser(self, laser_scan):
        self.laser_scan.append(laser_scan)
        self.data = True
        self.correcting = False

    def odom(self, data):
        if not self.extracted:
            self.extracted = True
            self.last_time = data.header.stamp
            self.last_data = data
            self.state[0], self.state[1] = data.pose.pose.position.x, data.pose.pose.position.y
            _ ,_ ,self.state[2] = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
            self.last_angle = self.state[2]
            self.broadcast_tf('odom', 'base_ekf', data.header.stamp, self.state[:3])
            #self.broadcast_tf('map', 'odom', data.header.stamp, self.state[:3])
        else:
            current_time = data.header.stamp
            _ ,_ ,new_yaw = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
            
            if abs(abs(data.pose.pose.position.x) - abs(self.last_data.pose.pose.position.x)) > 0.0001 or abs(abs(data.pose.pose.position.y) - abs(self.last_data.pose.pose.position.y)) > 0.0001 or abs(abs(new_yaw) - abs(self.last_angle)) > 0.0017:
                #dt = (current_time - self.last_time).to_sec()
                #odometry = [dt, data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.angular.z]
                
                delta_rot_1 = atan2(data.pose.pose.position.y - self.last_data.pose.pose.position.y, data.pose.pose.position.x - self.last_data.pose.pose.position.x) - self.last_angle
                delta_trans = np.sqrt((data.pose.pose.position.x - self.last_data.pose.pose.position.x)**2 + (data.pose.pose.position.y - self.last_data.pose.pose.position.y)**2)
                delta_rot_2 = new_yaw - self.last_angle - delta_rot_1
                odometry = [delta_rot_1, delta_trans, delta_rot_2]  
                
                self.state, self.covariance_matrix = self.filter.predict_step(self.state, self.covariance_matrix, odometry)
                

                print("ODOM STATE : " + str(self.state[:3]))
                self.moved = True
                
                self.broadcast_tf('odom', 'base_ekf', current_time, self.state[:3])
                #self.broadcast_tf('map', 'odom', current_time, self.state[:3] - np.array([data.pose.pose.position.x, data.pose.pose.position.y, new_yaw]))
    

                if self.laser_scan != []:
                    #self.broadcast_tf('odom', 'base_ekf', current_time, self.state[:3])
                    index = 0
                    for laser_scan in self.laser_scan:
                        if abs(current_time - laser_scan.header.stamp).to_sec() < 0.05:
                            if laser_scan.header.stamp > current_time:
                                break
                           
                            self.state, self.covariance_matrix = self.filter.correction_step(self.state, self.covariance_matrix, laser_scan, laser_scan.header.stamp)
                            self.laser_scan.pop(index)
                            self.filter.visualizer.publish_map(self.state, self.filter.landmarks, self.filter.feature_detection)
                            break
                        else:
                            self.laser_scan.pop(index)
                            print("DISCARTING DATA")
                        index += 1
    
            else:
                self.broadcast_tf('odom', 'base_ekf', current_time, self.state[:3])
                self.filter.visualizer.publish_map(self.state, self.filter.landmarks, self.filter.feature_detection)
            #self.broadcast_tf('map', 'odom', current_time, self.state[:3] - np.array([data.pose.pose.position.x, data.pose.pose.position.y, new_yaw]))
            
            self.filter.visualizer.publish_odom_path(data)
            self.filter.visualizer.publish_ekf_path(self.state[:3], data)
            self.last_data = data
            self.last_angle = new_yaw
            self.last_time = current_time
            
if __name__ == "__main__":
    rospy.init_node("odometry_node")
    sub = Sub("odom", "scan")
    rospy.spin()