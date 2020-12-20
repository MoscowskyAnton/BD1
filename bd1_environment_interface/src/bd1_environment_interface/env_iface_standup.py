#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from bd1_environment_interface.srv import SetAction

import numpy as np

class EnvIfaceStandUp(object):
    def __init__(self):
        
        self.name = "environment_interface_standup"
        rospy.init_node(self.name)
        
        # action interfaces for servos commands
        self.right_leg_client = actionlib.SimpleActionClient('right_leg_servo_states_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("[{}] waiting for right_leg_servo_states_controller action...".format(self.name))
        self.right_leg_client.wait_for_server()
        rospy.loginfo("[{}] right_leg_servo_states_controller action loaded".format(self.name))
        
        self.left_leg_client = actionlib.SimpleActionClient('left_leg_servo_states_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("[{}] waiting for left_leg_servo_states_controller action...".format(self.name))
        self.left_leg_client.wait_for_server()
        rospy.loginfo("[{}] left_leg_servo_states_controller action loaded".format(self.name))
        
        self.head_client = actionlib.SimpleActionClient('head_servo_state_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo("[{}] waiting for head_servo_states_controller action...".format(self.name))
        self.head_client.wait_for_server()
        rospy.loginfo("[{}] head_servo_states_controller action loaded".format(self.name))
        
        # gazebo
        rospy.wait_for_service('gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        
        rospy.Service("~reset", Empty, self.reset_cb)
        
        rospy.Service("~get_state_and_reward", Empty, self.get_state_and_reward_cb)
        
        rospy.Service("~set_action", SetAction, self.set_action_cb)
        
        rospy.logwarn("[{}] ready!".format(self.name))
        
    def reset_cb(self, req):    
        # conceal legs
        self.right_leg_client.send_goal(self.right_leg_cmd_pose(1.5,-3,1.5))
        self.left_leg_client.send_goal(self.left_leg_cmd_pose(1.5,-3,1.5))
        self.head_client.send_goal(self.head_cmd_pose(-1.5,1.5))
        self.right_leg_client.wait_for_result()
        self.left_leg_client.wait_for_result()
        self.head_client.wait_for_result()
        # replace robot 
        ms = ModelState()
        ms.model_name = "bd1"
        ms.pose.position.z = 0.5
        self.set_model_state_srv(ms)
        
        return []        
    
    def right_leg_cmd_pose(self, up, mid, feet):
        return self.leg_cmd_pose('r', up, mid, feet)
    
    def left_leg_cmd_pose(self, up, mid, feet):
        return self.leg_cmd_pose('l', up, mid, feet)
    
    def right_leg_cmd_vel(self, up, mid, feet):
        return self.leg_cmd_vel('r', up, mid, feet)
    
    def left_leg_cmd_vel(self, up, mid, feet):
        return self.leg_cmd_vel('l', up, mid, feet)
    
    def leg_cmd_pose(self, side, up, mid, feet):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ['up_leg_{}_j'.format(side), 'mid_leg_{}_j'.format(side), 'feet_{}_j'.format(side)]
        point = JointTrajectoryPoint()
        point.positions = [up, mid, feet]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
    
    def leg_cmd_vel(self, side, up, mid, feet):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ['up_leg_{}_j'.format(side), 'mid_leg_{}_j'.format(side), 'feet_{}_j'.format(side)]
        point = JointTrajectoryPoint()
        point.positions = [np.sign(up)*np.pi/2, np.sign(mid)*np.pi/2-np.pi/2, np.sign(feet)*np.pi/2]
        point.velocities = [up, mid, feet]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
    
    def head_cmd_pose(self, neck, head):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ["neck_j", "head_j"]
        point = JointTrajectoryPoint()
        point.positions = [neck, head]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
    
    def head_cmd_vel(self, neck, head):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ["neck_j", "head_j"]
        point = JointTrajectoryPoint()
        point.positions = [neck, head]
        point.velocities = [neck, head]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
            
    def get_state_and_reward_cb(self, req):
        pass
    
    def set_action_cb(self, req):
        self.right_leg_client.send_goal(self.right_leg_cmd_vel(req.up_r, req.mid_r, req.feet_r))
        self.left_leg_client.send_goal(self.left_leg_cmd_vel(req.up_l, req.mid_l, req.feet_l))
        #self.head_cmd_vel(req.neck, req.head)
        return []
        
    def run(self):
        rospy.spin()
        
if __name__ == '__main__' :
    eisu = EnvIfaceStandUp()
    eisu.run()
    
