#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint
from bd1_environment_interface.srv import SetAction, GetStateAndReward, GetStateAndRewardResponse
from bd1_environment_interface.msg import State
from tf.transformations import euler_from_quaternion
import numpy as np

class EnvIfaceStandUp(object):
    def __init__(self):
        
        self.name = "environment_interface_standup"
        rospy.init_node(self.name)
        
        self.target_z = rospy.get_param("~target_z", None)
        if self.target_z is None:
            rospy.logerr("[{}] target_z does not specified! Exit.".format(self.name))
            exit()
            
        self.episode_end = False
        
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
        
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        # Interface
        rospy.Service("~reset", Empty, self.reset_cb)
        
        rospy.Service("~get_state_and_reward", GetStateAndReward, self.get_state_and_reward_cb)
        
        rospy.Service("~set_action", SetAction, self.set_action_cb)
                
        
        # states
        self.right_leg_state = JointTrajectoryPoint()
        self.left_leg_state = JointTrajectoryPoint()
        self.head_state = JointTrajectoryPoint()        
        
        rospy.Subscriber("right_leg_servo_states_controller/state", JointTrajectoryControllerState, self.right_leg_state_cb)
        rospy.Subscriber("left_leg_servo_states_controller/state", JointTrajectoryControllerState, self.left_leg_state_cb)
        rospy.Subscriber("/head_servo_state_controller/state", JointTrajectoryControllerState, self.head_state_cb)
        
        rospy.logwarn("[{}] ready!".format(self.name))
        
    def right_leg_state_cb(self, msg):
        self.right_leg_state = msg.actual
        
    def left_leg_state_cb(self, msg):
        self.left_leg_state = msg.actual
        
    def head_state_cb(self, msg):
        self.head_state = msg.actual
        
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
        
        self.episode_end = False
        
        return []        
    
    def right_leg_cmd_pose(self, up, mid, feet):
        return self.leg_cmd_pose('r', up, mid, feet)
    
    def left_leg_cmd_pose(self, up, mid, feet):
        return self.leg_cmd_pose('l', up, mid, feet)        
    
    def leg_cmd_pose(self, side, up, mid, feet):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ['up_leg_{}_j'.format(side), 'mid_leg_{}_j'.format(side), 'feet_{}_j'.format(side)]
        point = JointTrajectoryPoint()
        point.positions = [up, mid, feet]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
    
    def leg_cmd_vel(self, side, up_p, mid_p, feet_p, up_v, mid_v, feet_v):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ['up_leg_{}_j'.format(side), 'mid_leg_{}_j'.format(side), 'feet_{}_j'.format(side)]
        point = JointTrajectoryPoint()
        point.positions = [up_p, mid_p, feet_p]
        point.velocities = [up_v, mid_v, feet_v]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
    
    def right_leg_cmd_vel(self, up_p, mid_p, feet_p, up_v, mid_v, feet_v):
        return self.leg_cmd_vel('r', up_p, mid_p, feet_p, up_v, mid_v, feet_v)
    
    def left_leg_cmd_vel(self, up_p, mid_p, feet_p, up_v, mid_v, feet_v):
        return self.leg_cmd_vel('l', up_p, mid_p, feet_p, up_v, mid_v, feet_v)
    
    def head_cmd_pose(self, neck, head):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ["neck_j", "head_j"]
        point = JointTrajectoryPoint()
        point.positions = [neck, head]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal
    
    def head_cmd_vel(self, neck_p, head_p, neck_v, head_v):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ["neck_j", "head_j"]
        point = JointTrajectoryPoint()
        point.positions = [neck_p, head_p]
        point.velocities = [neck_v, head_v]
        point.time_from_start = rospy.Duration(1)
        goal.trajectory.points.append(point)
        return goal                
    
    def set_action_cb(self, req):
        self.right_leg_client.send_goal(
            self.right_leg_cmd_vel(req.up_p_r, req.mid_p_r, req.feet_p_r,
                                   req.up_v_r, req.mid_v_r, req.feet_v_r))        
        self.left_leg_client.send_goal(
            self.left_leg_cmd_vel(req.up_p_l, req.mid_p_l, req.feet_p_l,
                                   req.up_v_l, req.mid_v_l, req.feet_v_l))        
            
        self.head_client.send_goal(self.head_cmd_vel(req.neck_p, req.head_p, req.neck_v, req.head_v))
        return []
        
    def get_state_and_reward_cb(self, req):
        res = GetStateAndRewardResponse()
        
        # MODEL STATE
        model_state = self.get_model_state_srv("bd1","")
        res.state.pose_x = model_state.pose.position.x
        res.state.pose_y = model_state.pose.position.y
        res.state.pose_z = model_state.pose.position.z
                                
        rpy = euler_from_quaternion([model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w])                
        res.state.rot_r = rpy[0]
        res.state.rot_p = rpy[1]
        res.state.rot_y = rpy[2]
        
        # REWARD
        res.reward = -np.power(self.target_z - res.state.pose_z, 2)
        
        # SERVOS POSITIONS        
        # right
        res.state.up_p_r = self.right_leg_state.positions[0]
        res.state.mid_p_r = self.right_leg_state.positions[1]
        res.state.feet_p_r = self.right_leg_state.positions[2]
        res.state.up_v_r = self.right_leg_state.velocities[0]
        res.state.mid_v_r = self.right_leg_state.velocities[1]
        res.state.feet_v_r = self.right_leg_state.velocities[2]
        # left
        res.state.up_p_l = self.left_leg_state.positions[0]
        res.state.mid_p_l = self.left_leg_state.positions[1]
        res.state.feet_p_l = self.left_leg_state.positions[2]
        res.state.up_v_l = self.left_leg_state.velocities[0]
        res.state.mid_v_l = self.left_leg_state.velocities[1]
        res.state.feet_v_l = self.left_leg_state.velocities[2]
        # head
        res.state.neck_p = self.head_state.positions[0]
        res.state.neck_v = self.head_state.velocities[0] 
        res.state.head_p = self.head_state.positions[1]
        res.state.head_v = self.head_state.velocities[1] 
                     
        # SIMPLE ROBOT FALL DETECTOR
        if np.absolute(res.state.rot_p) > 1.4:
            self.episode_end = True
        if np.absolute(res.state.rot_r) > 1.4:
            self.episode_end = True
            
        res.episode_end = self.episode_end
        return res
                
    def run(self):
        rospy.spin()
        
if __name__ == '__main__' :
    eisu = EnvIfaceStandUp()
    eisu.run()
    
