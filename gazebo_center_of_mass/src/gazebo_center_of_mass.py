#!/usr/bin/env python

import copy
import re
import threading
import xml
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import rospy

from geometry_msgs.msg import Point32, Point
from gazebo_msgs.msg import LinkStates
from visualization_msgs.msg import MarkerArray,Marker

re_xyz = re.compile(' *(.+) +(.+) +(.+) *')

class CalcCenterOfMass(object):
    def __init__(self):
        rospy.init_node('gazebo_center_of_mass')
        self.robot_name = rospy.get_param('~robot_name')
        self.debug_output = rospy.get_param('~debug_mode', True)
        self.load_robot_description()
        # publisher for center of mass
        loaded_descr_str = 'gazebo_center_of_mass: loaded links are:'
        for nm, (mass, pos) in self.robot_links:
            loaded_descr_str += '\n{}, m={}: {}'.format(nm, mass, pos)
        rospy.loginfo(loaded_descr_str)
        self.center_of_mass_pub = rospy.Publisher('center_of_mass', Point32, queue_size=1)
        self.out_msg = Point32()
        if self.debug_output:
            self.prepare_debug_msg()
            self.debug_pub = rospy.Publisher('debug_center_of_mass', MarkerArray, queue_size=1)

        # timer
        self.lock = threading.Lock()  # mutex
        self.curr_pos = {}
        self.timer_cb = rospy.Timer(rospy.Duration(0.1), self.timer_cb)
            
        # subscribe on gazebo data
        self.gazebo_state_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_cb)
        

    def link_states_cb(self, msg):
        """ read data about position of each link and save it in self.curr_pos """
        self.lock.acquire()
        self.curr_pos = {}
        for link_name,_ in self.robot_links:
            try:
                msg_ind = msg.name.index(link_name)
            except ValueError:
                continue
            self.curr_pos[link_name] = msg.pose[msg_ind]
        self.lock.release()


    def timer_cb(self, e):
        """ generate positions with fixed rate """
        self.lock.acquire()
        curr_pos = copy.copy(self.curr_pos)
        self.lock.release()
        self.out_msg.x = 0
        self.out_msg.y = 0
        self.out_msg.z = 0
        sum_mass = 0
        # read data
        for ind, (k, v) in enumerate(self.robot_links):
            if k not in curr_pos.keys():
                continue
            link_pose = curr_pos[k]
            # rotate position of the center of mass
            r = Rotation.from_quat([link_pose.orientation.x,
                                    link_pose.orientation.y,
                                    link_pose.orientation.z,
                                    link_pose.orientation.w]).as_dcm()
            pos = np.dot(r, np.matrix([[v[1][0]],
                                       [v[1][1]],
                                       [v[1][2]]]))
            # shift position
            self.out_msg.x += v[0] * (link_pose.position.x + pos[0][0])
            self.out_msg.y += v[0] * (link_pose.position.y + pos[1][0])
            self.out_msg.z += v[0] * (link_pose.position.z + pos[2][0])
            if self.debug_output:
                self.debug_change_link_center(ind,
                                              link_pose.position.x + pos[0][0],
                                              link_pose.position.y + pos[1][0],
                                              link_pose.position.z + pos[2][0])
            sum_mass += v[0]
        if sum_mass < 1e-6:
            return # massive objects not found
        self.out_msg.x /= sum_mass
        self.out_msg.y /= sum_mass
        self.out_msg.z /= sum_mass
        self.center_of_mass_pub.publish(self.out_msg)
        # debug output
        if self.debug_output:
            self.debug_change_full_center(self.out_msg.x, self.out_msg.y, self.out_msg.z)
            self.debug_pub.publish(self.debug_msg)
        #rospy.logerr(self.out_msg)
        
    
    def load_robot_description(self):
        descr_xml_src = rospy.get_param('robot_description')
        root = ET.fromstring(descr_xml_src)
        self.robot_links = {}
        if root.tag == 'robot':
            for ch1 in root:
                if ch1.tag == 'link':
                    link_name = self.robot_name + '::' + ch1.attrib['name']
                    mass = 0
                    origin = (0, 0, 0)
                    for ch2 in ch1:
                        if ch2.tag == 'inertial':
                            for ch3 in ch2:
                                if ch3.tag == 'mass':
                                    mass = float(ch3.attrib['value'])
                                if ch3.tag == 'origin':
                                    origin_str = ch3.attrib['xyz']
                                    m = re_xyz.match(origin_str)
                                    if m is not None:
                                        origin = (float(m.group(1)),
                                                  float(m.group(2)),
                                                  float(m.group(3)))
                    self.robot_links[link_name] = (mass, origin)
        # convert map to list
        self.robot_links = self.robot_links.items()

    def prepare_debug_msg(self):
        self.debug_msg = MarkerArray()
        # first element is the total center of mass
        marker_center = Marker()
        marker_center.header.frame_id = 'map'
        marker_center.id = 0
        marker_center.type = Marker.CUBE
        marker_center.action = 0
        marker_center.scale.x = 0.05
        marker_center.scale.y = 0.05
        marker_center.scale.z = 0.05
        marker_center.color.r = 1
        marker_center.color.g = 0
        marker_center.color.b = 0
        marker_center.color.a = 1
        self.debug_msg.markers.append(marker_center)
        # first element is the total center of mass
        marker_links = Marker()
        marker_links.header.frame_id = 'map'
        marker_links.id = 1
        marker_links.type = Marker.POINTS
        marker_links.action = 0
        marker_links.scale.x = 0.05
        marker_links.scale.y = 0.05
        marker_links.color.r = 0
        marker_links.color.g = 1
        marker_links.color.b = 0
        marker_links.color.a = 1
        for i in range(len(self.robot_links)):
            marker_links.points.append(Point())
        self.debug_msg.markers.append(marker_links)

    def debug_change_full_center(self, x, y, z):
        self.debug_msg.markers[0].pose.position.x = x
        self.debug_msg.markers[0].pose.position.y = y
        self.debug_msg.markers[0].pose.position.z = z
        self.debug_msg.markers[0].pose.orientation.x = 0
        self.debug_msg.markers[0].pose.orientation.y = 0
        self.debug_msg.markers[0].pose.orientation.z = 0
        self.debug_msg.markers[0].pose.orientation.w = 1

    def debug_change_link_center(self, ind, x, y, z):
        self.debug_msg.markers[1].points[ind].x = x
        self.debug_msg.markers[1].points[ind].y = y
        self.debug_msg.markers[1].points[ind].z = z

if __name__ == '__main__':
    center_of_mass = CalcCenterOfMass()
    rospy.spin()
