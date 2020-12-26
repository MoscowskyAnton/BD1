#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import Bool, Float64
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint
from bd1_environment_interface.srv import SetAction, SetVectAction, GetStateAndReward, GetStateAndRewardResponse, GetVectStateAndReward, GetVectStateAndRewardResponse
from bd1_environment_interface.msg import State
from tf.transformations import euler_from_quaternion
import numpy as np
import tf2_ros
from sensor_msgs.msg import JointState

class CircleBuffer(object):
    def __init__(self, max_el):
        self.max_el = max_el
        self.storage = []
        self.pos = 0
        
    def append(self, x):
        if len(self.storage) < self.max_el:
            self.storage.append(x)
        else:
            self.storage[self.pos] = x
            self.pos+=1
            if self.pos >= self.max_el:
                self.pos = 0
                
    def in_(self, x):
        return x in self.storage

# un norm from [-1; 1]
def unnorm(x, x_min, x_max):
    #return x_min + x * (x_max-x_min)
    return ((x+1)/2)*(x_max-x_min)  + x_min

class EnvIfaceStandUp(object):
    def __init__(self):
        
        self.name = "environment_interface_standup"
        rospy.init_node(self.name)
        
        self.servo_control = rospy.get_param("~servo_control", 'VEL')
        
        self.target_x = rospy.get_param("~target_x", None)
        self.target_y = rospy.get_param("~target_y", None)
        self.target_z = rospy.get_param("~target_z", None)
        
        if self.target_x is None:
            rospy.logerr("[{}] target_x does not specified! Exit.".format(self.name))
            exit()
        if self.target_y is None:
            rospy.logerr("[{}] target_y does not specified! Exit.".format(self.name))
            exit()            
        if self.target_z is None:
            rospy.logerr("[{}] target_z does not specified! Exit.".format(self.name))
            exit()            
            
        ## 
        # TODO to params, or better read somehow from urdf or whatever
        self.max_vel_servo = 1
        self.max_feet_p = 1.5#np.pi/2
        self.min_feet_p = -np.pi/2
        self.max_mid_p = 0
        self.min_mid_p = 3#-np.pi
        self.max_up_p = 1.5#np.pi/2
        self.min_up_p = -np.pi/2        
        self.max_head_p = 1.5#np.pi/2
        self.min_head_p = -1.5#-np.pi/2
        ##
            
        self.falls = CircleBuffer(20)
        self.episode_end = False
        
        if self.servo_control == 'TRAJ':
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
            
            # states
            self.right_leg_state = JointTrajectoryPoint()
            self.left_leg_state = JointTrajectoryPoint()
            self.head_state = JointTrajectoryPoint()        
            
            rospy.Subscriber("right_leg_servo_states_controller/state", JointTrajectoryControllerState, self.right_leg_state_cb)
            rospy.Subscriber("left_leg_servo_states_controller/state", JointTrajectoryControllerState, self.left_leg_state_cb)
            rospy.Subscriber("/head_servo_state_controller/state", JointTrajectoryControllerState, self.head_state_cb)
            
        if self.servo_control == 'VEL':
            
            self.head_pub = rospy.Publisher('head_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.neck_pub = rospy.Publisher('neck_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.up_r_pub = rospy.Publisher('leg_up_r_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.mid_r_pub = rospy.Publisher('leg_mid_r_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.feet_r_pub = rospy.Publisher('feet_r_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.up_l_pub = rospy.Publisher('leg_up_l_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.mid_l_pub = rospy.Publisher('leg_mid_l_servo_velocity_controller/command', Float64, queue_size = 1)
            
            self.feet_l_pub = rospy.Publisher('feet_l_servo_velocity_controller/command', Float64, queue_size = 1)
            
            # robot state            
            #self.parent_link = "base_link"
            #self.links = ['neck_link', 'head_link', 'up_leg_r_link', 'mid_leg_r_link', 'feet_r_link', 'up_leg_l_link', 'mid_leg_l_link', 'feet_l_link']
            #self.tfBuffer = tf2_ros.Buffer()
            #self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
            self.last_joint_states = None
            rospy.Subscriber("joint_states", JointState, self.joint_states_cb)
            
                
        # gazebo
        rospy.wait_for_service('gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        rospy.wait_for_service('gazebo/reset_world')
        self.reset_world_srv = rospy.ServiceProxy('gazebo/reset_world', Empty)
        
        # Interface
        rospy.Service("~reset", Empty, self.reset_cb)
        
        rospy.Service("~get_state_and_reward", GetStateAndReward, self.get_state_and_reward_cb)
        
        rospy.Service("~get_vect_state_and_reward", GetVectStateAndReward, self.get_vect_state_and_reward_cb)
        
        rospy.Service("~set_action", SetAction, self.set_action_cb)
        
        rospy.Service("~set_vect_action", SetVectAction, self.set_vect_action_cb)                               
        
        self.last_fall = False
        rospy.Subscriber("fall_detector/fall", Bool, self.fall_cb)
                
        rospy.logwarn("[{}] ready!".format(self.name))
        
    def joint_states_cb(self, msg):
        self.last_joint_states = msg
        
    def fall_cb(self, msg):
        #self.last_fall = msg.data
        self.falls.append(msg.data)
        
    def right_leg_state_cb(self, msg):
        self.right_leg_state = msg.actual
        
    def left_leg_state_cb(self, msg):
        self.left_leg_state = msg.actual
        
    def head_state_cb(self, msg):
        self.head_state = msg.actual
        
    def reset_cb(self, req):
        #self.reset_world_srv()
        # conceal legs        
        if self.servo_control == 'TRAJ':            
            self.right_leg_client.send_goal(self.right_leg_cmd_pose(1.5,-3,1.5))
            self.left_leg_client.send_goal(self.left_leg_cmd_pose(1.5,-3,1.5))
            self.head_client.send_goal(self.head_cmd_pose(-1.5,1.5))
            self.right_leg_client.wait_for_result()
            self.left_leg_client.wait_for_result()
            self.head_client.wait_for_result()
            
        if self.servo_control == 'VEL':
            ms = ModelState()
            ms.model_name = "bd1"
            ms.pose.position.z = 30
            self.set_model_state_srv(ms)
            
            self.send_vel_cmd_left_leg(self.max_vel_servo, -self.max_vel_servo, self.max_vel_servo)
            self.send_vel_cmd_right_leg(self.max_vel_servo, -self.max_vel_servo, self.max_vel_servo)
            self.send_vel_cmd_head(-self.max_vel_servo, self.max_vel_servo)
            rospy.sleep(2.5) # TODO place for improovment
                
            #self.send_vel_cmd_left_leg(0, 0, 0)
            #self.send_vel_cmd_right_leg(0, 0, 0)
            #self.send_vel_cmd_head(0, 0)
            
        # replace robot 
        ms = ModelState()
        ms.model_name = "bd1"
        ms.pose.position.z = 0.15
        self.set_model_state_srv(ms)
        rospy.sleep(0.5)        
        
        
        
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
    
    def send_vel_cmd_left_leg(self, up, mid, feet):
        self.feet_l_pub.publish(feet)        
        self.mid_l_pub.publish(mid)    
        self.up_l_pub.publish(up)
    
    def send_vel_cmd_right_leg(self, up, mid, feet):
        self.feet_r_pub.publish(feet)        
        self.mid_r_pub.publish(mid)    
        self.up_r_pub.publish(up)
        
    def send_vel_cmd_head(self, neck, head):
        self.head_pub.publish(head)
        self.neck_pub.publish(neck)
                    
    def set_vect_action_cb(self, req):
        # unvector & unnormilize it
        va = req.vector_action
        
        if self.servo_control == 'TRAJ':
            up_p_r = unnorm(va[0], self.min_up_p, self.max_up_p)
            mid_p_r = unnorm(va[1], self.min_mid_p, self.max_mid_p)        
            feet_p_r = unnorm(va[2], self.min_feet_p, self.max_feet_p)
            #up_p_l = unnorm(va[3], self.min_up_p, self.max_up_p)
            #mid_p_l = unnorm(va[4], self.min_mid_p, self.max_mid_p)
            #feet_p_l = unnorm(va[5], self.min_feet_p, self.max_feet_p)
            up_p_l = up_p_r
            mid_p_l = mid_p_r
            feet_p_l = feet_p_r
            rospy.logerr("feet {}".format(feet_p_l))
            
            neck_p = unnorm(va[6], self.min_head_p, self.max_head_p)
            head_p = unnorm(va[7], self.min_head_p, self.max_head_p)
            
            # send
            self.right_leg_client.send_goal(
                self.right_leg_cmd_pose(up_p_r, mid_p_r, feet_p_r))        
            self.left_leg_client.send_goal(
                self.left_leg_cmd_pose(up_p_l, mid_p_l, feet_p_l))        
            
        # NOTE wait for result?
        if self.servo_control == 'VEL':
            up_v_r = unnorm(va[0], -self.max_vel_servo, self.max_vel_servo)
            mid_v_r = unnorm(va[1], -self.max_vel_servo, self.max_vel_servo)
            feet_r = unnorm(va[2], -self.max_vel_servo, self.max_vel_servo)
            
            #up_v_l = unnorm(va[3], -self.max_vel_servo, self.max_vel_servo)
            #mid_v_l = unnorm(va[4], -self.max_vel_servo, self.max_vel_servo)
            #feet_l = unnorm(va[5], -self.max_vel_servo, self.max_vel_servo)
            up_v_l = up_v_r
            mid_v_l = mid_v_r
            feet_l = feet_r
            
            neck_v = unnorm(va[6], -self.max_vel_servo, self.max_vel_servo)
            head_v = unnorm(va[7], -self.max_vel_servo, self.max_vel_servo)
            
            #self.feet_l_pub.publish(feet_l)
            #self.feet_r_pub.publish(feet_r)
            #self.mid_l_pub.publish(mid_v_l)
            #self.mid_r_pub.publish(mid_v_r)
            #self.up_l_pub.publish(up_v_l)
            #self.up_r_pub.publish(up_v_r)
            #self.head_pub.publish(head_v)
            #self.neck_pub.publish(neck_v)
            self.send_vel_cmd_left_leg(up_v_l, mid_v_l, feet_l)
            self.send_vel_cmd_right_leg(up_v_r, mid_v_r, feet_r)
            self.send_vel_cmd_head(neck_v, head_v)
            
            
        #self.head_client.send_goal(self.head_cmd_pose(neck_p, head_p))
        return []
    
    def get_vect_state_and_reward_cb(self, req):
        res = GetVectStateAndRewardResponse()
        
        # MODEL STATE
        model_state = self.get_model_state_srv("bd1","")
        res.state[0] = model_state.pose.position.x
        res.state[1] = model_state.pose.position.y
        res.state[2] = model_state.pose.position.z
                                
        rpy = euler_from_quaternion([model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w])                
        res.state[3] = rpy[0]
        res.state[4] = rpy[1]
        res.state[5] = rpy[2]
        
        # SERVOS positions
        #for i, link in enumerate(self.links):
            #trans = self.tfBuffer.lookup_transform(self.parent_link, link, rospy.Time())
            #res.state[6+i] = euler_from_quaternion([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])[1]                
        pos_vel = [self.last_joint_states.position[0], self.last_joint_states.position[3],self.last_joint_states.position[6],self.last_joint_states.velocity[0], self.last_joint_states.velocity[3],self.last_joint_states.velocity[6]]    
        
        res.state[6:12] = pos_vel
            
            
        # REWARD
        #res.reward = - np.power(self.target_z - model_state.pose.position.z, 2) 
        
        #res.reward = -( np.power(self.target_x - model_state.pose.position.x, 2) + np.power(self.target_y - model_state.pose.position.y, 2) + np.power(self.target_z - model_state.pose.position.z, 2) )
        #res.reward = -( np.power(self.target_z - model_state.pose.position.z, 2) + np.absolute(np.sin(rpy[1])) )
        res.reward = model_state.pose.position.z**2
            
        res.episode_end = self.falls.in_(True)
        
        return res
            
        
        
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
        res.reward = -( np.power(self.target_x - res.state.pose_x, 2) + np.power(self.target_y - res.state.pose_y, 2) + np.power(self.target_z - res.state.pose_z, 2) )
        
        # SERVOS POSITIONS        
        # right
        res.state.up_p_r = self.right_leg_state.positions[0]
        res.state.mid_p_r = self.right_leg_state.positions[1]
        res.state.feet_p_r = self.right_leg_state.positions[2]
        #res.state.up_v_r = self.right_leg_state.velocities[0]
        #res.state.mid_v_r = self.right_leg_state.velocities[1]
        #res.state.feet_v_r = self.right_leg_state.velocities[2]
        # left
        res.state.up_p_l = self.left_leg_state.positions[0]
        res.state.mid_p_l = self.left_leg_state.positions[1]
        res.state.feet_p_l = self.left_leg_state.positions[2]
        #res.state.up_v_l = self.left_leg_state.velocities[0]
        #res.state.mid_v_l = self.left_leg_state.velocities[1]
        #res.state.feet_v_l = self.left_leg_state.velocities[2]
        # head
        res.state.neck_p = self.head_state.positions[0]
        
        #res.state.neck_v = self.head_state.velocities[0] 
        res.state.head_p = self.head_state.positions[1]
        #res.state.head_v = self.head_state.velocities[1]                              
            
        res.episode_end = self.last_fall
        return res
                
    def run(self):
        rospy.spin()
        
if __name__ == '__main__' :
    eisu = EnvIfaceStandUp()
    eisu.run()
    
