import rospy
import tf2_ros
import numpy as np
from math import sin, cos 
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler

class RvizMarker:
    def __init__(self) -> None:
        self.rate = rospy.Rate(10)
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(4))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.odom_path_publisher = rospy.Publisher("/odom_path", Path, queue_size=1)
        self.ekf_path_publisher = rospy.Publisher("/ekf_path", Path, queue_size=1)
        self.marker_publisher = rospy.Publisher("/marker_basic", Marker, queue_size=1)
        self.marker_index = 0
        self.odom_path = Path()
        self.ekf_path = Path()

    def polar_to_cartesian(self, predicted_measurment):
        r, p = predicted_measurment
        return [r * cos(p), r*sin(p)]
    
    def publish_map(self, state, landmarks, feature_object):
        if landmarks == []:
            return
        points = np.array([[0,0],[0,0]], dtype=np.float64)
        for landmark_index in range(0, len(landmarks)):
            landmark_state = state[2*landmark_index + 3: 2*landmark_index + 5]
            point = self.polar_to_cartesian(landmark_state)
            #self.publish_point(point, 1, 0, 1)
            m = -point[0] / point[1]
            b = point[1] - m*point[0]
            copy_landmarks = landmarks[landmark_index].copy()
            for point in copy_landmarks:
                proj = feature_object.projection_point2line(point, m, b)
                point[0] = proj[0]
                point[1] = proj[1]
            
            points = np.append(points, copy_landmarks, axis=0)

        self.publish_line(points, 0)
           
             
    def publish_line(self, line_points, id = -1):
        line_marker = Marker()
        line_marker.action = line_marker.ADD
        
        line_marker.header.frame_id = "map"
        line_marker.id = self.marker_index if id == -1 else id
        line_marker.type = line_marker.LINE_STRIP

        line_marker.scale.x = 0.01

        line_marker.color.r = 0.0
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0

        line_marker.color.a = 1

        line_marker.pose.orientation.x = 0.0
        line_marker.pose.orientation.y = 0.0
        line_marker.pose.orientation.z = 0.0
        line_marker.pose.orientation.w = 1.0

        line_marker.lifetime = rospy.Duration(0)

        for point in line_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0
            
            line_marker.points.append(p)
            if len(line_marker.points) == 2:
                self.marker_publisher.publish(line_marker)
                line_marker.points.pop()
                line_marker.points.pop()

                self.marker_index += 1
                line_marker.id = self.marker_index if id == -1 else line_marker.id+1

    def publish_point(self, point, color_r = 0.0, color_g = 1.0, color_b = 0.0):
        points = Marker()

        points.header.frame_id = "map"
        points.id = self.marker_index + 1
        points.type = points.POINTS

        points.scale.x = 0.1
        points.scale.y = 0.1

        points.color.r = color_r
        points.color.g = color_g
        points.color.b = color_b

        points.color.a = 1
        points.lifetime = rospy.Duration(1)

        points.pose.orientation.x = 0.0
        points.pose.orientation.y = 0.0
        points.pose.orientation.z = 0.0
        points.pose.orientation.w = 1.0

        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = 0

        points.points.append(p)
        self.marker_publisher.publish(points)

        self.marker_index += 1

    def publish_laser_points(self, laser_points):
        points = Marker()

        points.action = points.ADD


        points.header.frame_id = "map"
        points.id = self.marker_index + 1
        points.type = points.POINTS

        points.scale.x = 0.005
        points.scale.y = 0.005


        points.color.r = 0.0
        points.color.g = 1.0
        points.color.b = 0.0

        points.color.a = 1
       
        points.pose.orientation.x = 0.0
        points.pose.orientation.y = 0.0
        points.pose.orientation.z = 0.0
        points.pose.orientation.w = 1.0
        
        
        points.lifetime = rospy.Duration(1)

        for point in laser_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0

            points.points.append(p)

        self.marker_publisher.publish(points)
        self.marker_index += 1


    def publish_odom_path(self, data):
        self.odom_path.header = data.header
        pose = PoseStamped()
        pose.header = data.header
        pose.pose = data.pose.pose
        self.odom_path.poses.append(pose)
        self.odom_path_publisher.publish(self.odom_path)

    def publish_ekf_path(self, state, data):
        self.ekf_path.header = data.header
        pose = PoseStamped()
        pose.header = data.header
        pose.pose.position.x = state[0]
        pose.pose.position.y = state[1]
        pose.pose.position.z = 0
        quaternion = quaternion_from_euler(0, 0, state[2])
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        self.ekf_path.poses.append(pose)
        self.ekf_path_publisher.publish(self.ekf_path)



