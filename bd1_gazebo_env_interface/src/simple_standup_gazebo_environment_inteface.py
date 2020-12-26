#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
import numpy as np
from bd1_gazebo_env_interface.srv import Step, Reset, StepResponse, ResetResponse
from tf.transformations import euler_from_quaternion

def unnorm(x, x_min, x_max):    
        return ((x+1)/2)*(x_max-x_min)  + x_min

class UniversalGazeboEnvironmentInterface(object):
    def __init__(self):
        
        rospy.init_node('simple_standup_gazebo_environment_interface')
        
        self.name = rospy.get_name()
        
        self.max_velocity_lim = 1.0
        
        # service clients
        rospy.wait_for_service('gazebo/reset_simulation')
        self.reset_sim_srv = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        # publishers
        self.head_pub = rospy.Publisher('head_servo_velocity_controller/command', Float64, queue_size = 1)
            
        self.neck_pub = rospy.Publisher('neck_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.up_r_pub = rospy.Publisher('leg_up_r_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.mid_r_pub = rospy.Publisher('leg_mid_r_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.feet_r_pub = rospy.Publisher('feet_r_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.up_l_pub = rospy.Publisher('leg_up_l_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.mid_l_pub = rospy.Publisher('leg_mid_l_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.feet_l_pub = rospy.Publisher('feet_l_servo_velocity_controller/command', Float64, queue_size = 1)
        
        # subscribers and data containers
        self.last_joint_states = None
        rospy.Subscriber("joint_states", JointState, self.joint_states_cb)
        
        self.last_episode_fall = []
        rospy.Subscriber("fall_detector/fall", Bool, self.fall_cb)
        
        # service servers
        rospy.Service("~reset", Reset, self.reset_cb)
        rospy.Service("~step", Step, self.step_cb)
        
        rospy.logwarn("[{}] ready!".format(self.name))
    
    #
    # HIGH INTERFACE FUNCTIONS
    #
    def step_cb(self, req):
        # clear fall data        
        self.last_episode_fall = []
        self.set_action(req.action)
        rospy.sleep(req.step_duration_sec)
        state = self.get_state()
        return StepResponse(state, self.get_reward(state), self.check_done())
    
    def reset_cb(self, req):
        self.pub_action(0., 0., 0.)
        self.reset_sim_srv()        
        return ResetResponse(self.get_state())
    
    #
    # LOW INTERFACE FUNCTIONS
    #    
    def set_action(self, action):
        feet_v = unnorm(action[0], -self.max_velocity_lim, self.max_velocity_lim)        
        mid_v = unnorm(action[1], -self.max_velocity_lim, self.max_velocity_lim)
        up_v = unnorm(action[2], -self.max_velocity_lim, self.max_velocity_lim)
        self.pub_action(feet_v, mid_v, up_v)
        
    def pub_action(self, feet_v, mid_v, up_v):
        self.feet_l_pub.publish(feet_v)
        self.feet_r_pub.publish(feet_v)
        self.mid_l_pub.publish(mid_v)
        self.mid_r_pub.publish(mid_v)
        self.up_l_pub.publish(up_v)
        self.up_r_pub.publish(up_v)        
    
    def check_done(self):
        return True in self.last_episode_fall
    
    def get_reward(self, state):
        return -(0.3 - state[2])**2        
    
    def get_state(self):
        
        state = []
        
        # Model State        
        model_state = self.get_model_state_srv("bd1","")
        state.append(model_state.pose.position.x)
        state.append(model_state.pose.position.y)
        state.append(model_state.pose.position.z)
        
        rpy = euler_from_quaternion([model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w])                
        state += rpy
        
        # Joints Positions
        state.append( self.last_joint_states.position[0] )
        state.append( self.last_joint_states.position[3] )
        state.append( self.last_joint_states.position[6] )
        
        # Joint Velocities
        state.append( self.last_joint_states.velocity[0] )
        state.append( self.last_joint_states.velocity[3] )
        state.append( self.last_joint_states.velocity[6] )        
        
        return state               
        
    #
    # DATA COLLECTING
    #
    def joint_states_cb(self, msg):
        self.last_joint_states = msg
        
    def fall_cb(self, msg):
        self.last_episode_fall.append(msg.data)
        
    def run(self):
        rospy.spin()
        
if __name__ == '__main__' :
    ugei = UniversalGazeboEnvironmentInterface()
    ugei.run()
