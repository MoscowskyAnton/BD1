#!/usr/bin/env python
# coding: utf-8

from gazebo_msgs.msg import LinkState
from geometry_msgs.msg import Pose

def pose_to_dict(pose):
    p_dict = {}
    p_dict['position'] = {}
    p_dict['position']['x'] = pose.position.x
    p_dict['position']['y'] = pose.position.y
    p_dict['position']['z'] = pose.position.z
    p_dict['orientation'] = {}
    p_dict['orientation']['x'] = pose.orientation.x
    p_dict['orientation']['y'] = pose.orientation.y
    p_dict['orientation']['z'] = pose.orientation.z
    p_dict['orientation']['w'] = pose.orientation.w
    return p_dict

def link_state_to_dict(link_state):
    s_dict = {}
    s_dict['link_name'] = link_state.link_name
    s_dict['reference_frame'] = link_state.reference_frame
    s_dict['pose'] = pose_to_dict(link_state.pose)
    return s_dict

def dict_to_pose(p_dict):
    pose = Pose()
    pose.position.x = p_dict['position']['x'] 
    pose.position.y = p_dict['position']['y']
    pose.position.z = p_dict['position']['z']
    pose.orientation.x = p_dict['orientation']['x']
    pose.orientation.y = p_dict['orientation']['y'] 
    pose.orientation.z = p_dict['orientation']['z']
    pose.orientation.w = p_dict['orientation']['w']
    return pose

def dict_to_link_state(s_dict):
    link_state = LinkState()
    link_state.link_name = s_dict['link_name']
    link_state.reference_frame = s_dict['reference_frame']
    link_state.pose = dict_to_pose(s_dict['pose'])
    return link_state
