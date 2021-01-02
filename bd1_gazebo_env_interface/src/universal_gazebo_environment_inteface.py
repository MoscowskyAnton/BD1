#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
import numpy as np
from bd1_gazebo_env_interface.srv import Step, Reset, StepResponse, ResetResponse, Configure, ConfigureResponse
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import LinkStates
from bd1_gazebo_utils.msg import FeetContacts

def unnorm(x, x_min, x_max):    
        return ((x+1)/2)*(x_max-x_min)  + x_min

class UniversalGazeboEnvironmentInterface(object):
    def __init__(self):
        
        rospy.init_node('universal_gazebo_environment_interface')
        
        self.name = rospy.get_name()
        
        self.servo_control = rospy.get_param("~servo_control", "VEL")
        
        if self.servo_control == "VEL":
            self.max_action_lim = rospy.get_param("~max_servo_vel", 1.0)
        elif self.servo_control == "EFF":
            self.max_action_lim = rospy.get_param("~max_servo_eff", 30)
        else:
            rospy.logerr("[{}] unsupported servo control type {}! Exit.".format(self.name, self.servo_control))
        
        # service clients
        rospy.wait_for_service('gazebo/reset_simulation')
        self.reset_sim_srv = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        # publishers
        if self.servo_control == "VEL":
            self.head_pub = rospy.Publisher('head_servo_velocity_controller/command', Float64, queue_size = 1)                
            self.neck_pub = rospy.Publisher('neck_servo_velocity_controller/command', Float64, queue_size = 1)            
            self.hip_r_pub = rospy.Publisher('hip_r_servo_velocity_controller/command', Float64, queue_size = 1)            
            self.knee_r_pub = rospy.Publisher('knee_r_servo_velocity_controller/command', Float64, queue_size = 1)            
            self.foot_r_pub = rospy.Publisher('foot_r_servo_velocity_controller/command', Float64, queue_size = 1)            
            self.hip_l_pub = rospy.Publisher('hip_l_servo_velocity_controller/command', Float64, queue_size = 1)            
            self.knee_l_pub = rospy.Publisher('knee_l_servo_velocity_controller/command', Float64, queue_size = 1)            
            self.foot_l_pub = rospy.Publisher('foot_l_servo_velocity_controller/command', Float64, queue_size = 1)
        elif self.servo_control == "EFF":
            self.head_pub = rospy.Publisher('head_servo_effort_controller/command', Float64, queue_size = 1)                
            self.neck_pub = rospy.Publisher('neck_servo_effort_controller/command', Float64, queue_size = 1)            
            self.hip_r_pub = rospy.Publisher('hip_r_servo_effort_controller/command', Float64, queue_size = 1)            
            self.knee_r_pub = rospy.Publisher('knee_r_servo_effort_controller/command', Float64, queue_size = 1)            
            self.foot_r_pub = rospy.Publisher('foot_r_servo_effort_controller/command', Float64, queue_size = 1)            
            self.hip_l_pub = rospy.Publisher('hip_l_servo_effort_controller/command', Float64, queue_size = 1)            
            self.knee_l_pub = rospy.Publisher('knee_l_servo_effort_controller/command', Float64, queue_size = 1)            
            self.foot_l_pub = rospy.Publisher('foot_l_servo_effort_controller/command', Float64, queue_size = 1)
        
        # subscribers and data containers
        self.last_joint_states = None
        rospy.Subscriber("joint_states", JointState, self.joint_states_cb)
        
        self.last_episode_fall = []
        rospy.Subscriber("contacts_handler/fall", Bool, self.fall_cb)
        
        self.last_feet_contacts = None
        rospy.Subscriber("contacts_handler/feet_contacts", FeetContacts, self.feet_contacts_cb)
        
        self.last_mass_center = None
        rospy.Subscriber("center_of_mass", PointStamped, self.com_cb)
        self.last_press_center = None
        rospy.Subscriber("center_of_pressure_raw", PointStamped, self.cop_cb)
        self.last_link_states = None
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_cb)
        
        rospy.sleep(2) # KOSTYL
        
        self.state_types = {"base_pose": 3,
                            #"base_rot_rpy": 3,
                            "base_rot_quat":4,
                            "base_twist_lin":3,
                            "base_twist_ang":3,
                            #"left_leg_pos":3,
                            #"left_leg_pos_norm":6,
                            #"left_leg_vel":3,
                            #"right_leg_pos":3,
                            #"right_leg_pos_norm":6,
                            #"right_leg_vel":3,
                            #"all_head_pos":2,
                            #"all_head_pos_norm":4,
                            #"all_head_vel":2,
                            "com_abs":3,
                            "cop_abs":3,
                            "head_rot_quat":4,
                            "left_leg_all_quats":12,
                            "right_leg_all_quats":12,
                            "feet_contacts":4}
        
        self.actions_types = {"sync_legs_vel":3,
                             "left_legs_vel":3,
                             "right_legs_vel":3,
                             "all_head_vel":2}
        
        self.reward_types = {"stup_reward_z_1": self.stup_reward_z_1,
                             "stup_reward_z_2": self.stup_reward_z_2,
                             "stup_reward_z_3": self.stup_reward_z_3,
                             "stup_reward_z_contacts_1":self.stup_reward_z_contacts_1,
                             "stup_reward_z_contacts_2":self.stup_reward_z_contacts_2,
                             "stup_reward_z_contacts_3":self.stup_reward_z_contacts_3,
                             "stup_reward_z_contacts_4":self.stup_reward_z_contacts_4,
                             "stup_reward_z_pitch_1":self.stup_reward_z_pitch_1,
                             "stup_reward_z_pitch_2":self.stup_reward_z_pitch_2,
                             "stup_reward_z_pitch_vel_1": self.stup_reward_z_pitch_vel_1,
                             "stup_reward_z_com_cop_1": self.stup_reward_z_com_cop_1,
                             "stup_reward_z_pitch_com_cop_1": self.stup_reward_z_pitch_com_cop_1,
                             "stup_reward_z_body_pitch_com_cop_head_pitch_1": self.stup_reward_z_body_pitch_com_cop_head_pitch_1,
                             "stup_reward_z_body_pitch_com_cop_head_pitch_2": self.stup_reward_z_body_pitch_com_cop_head_pitch_2,
                             "stup_reward_z_body_pitch_com_cop_head_pitch_3":
                                 self.stup_reward_z_body_pitch_com_cop_head_pitch_3,
                                 "just_fall": self.just_fall_reward,
                                 "just_z": self.just_Z_reward,
                                 "max_z_and_body_pitch_1":self.max_z_and_body_pitch_1,
                                 "stup_reward_z_fall_penalty_1":self.stup_reward_z_fall_penalty_1,
                                 "walk_reward_max_vx":self.walk_reward_max_vx}
        
        self.requested_state = []
        self.requested_actions = []
        self.requested_reward = None
        self.configured = False
        self.state_dim = 0
        self.actions_dim = 0
        self.config_srv = rospy.Service("~configure", Configure, self.config_cb)
        rospy.logwarn("[{}] awaiting configuration...".format(self.name))                        
        
    
    #
    # HIGH INTERFACE FUNCTIONS
    #
    def config_cb(self, req):
        if not self.configured:
            self.state_dim = 0
            self.actions_dim = 0
            #rospy.logwarn("[{}] got config.".format(self.name))
            
            for state_el in req.state:
                if state_el in self.state_types:
                    self.state_dim += self.state_types[state_el]                    
                    self.requested_state.append(state_el)
                else:
                    rospy.logerr("[{}] unknown state element {} skipped!".format(self.name, state_el))                
                                    
            for action_el in req.actions:
                if action_el in self.actions_types:
                    self.actions_dim += self.actions_types[action_el]
                    self.requested_actions.append(action_el)
                else:
                    rospy.logerr("[{}] unknown action element {} skipped!".format(self.name, action_el))                                          
            
            if self.state_dim == 0:
                rospy.logerr("[{}] state vector is zero! Interface isn't configured, try again.".format(self.name))                                
                return ConfigureResponse(False, 0, 0, "", 0)
                
            if self.actions_dim == 0:
                rospy.logerr("[{}] action vector is zero! Interface isn't configured, try again.".format(self.name))                                
                return ConfigureResponse(False, 0, 0, "", 0)
            
            if req.reward in self.reward_types:
                self.requested_reward = self.reward_types[req.reward]
            else:
                rospy.logerr("[{}] {} reward not found! Interface isn't configured, try again.".format(self.name, req.reward))                                
                return ConfigureResponse(False, 0, 0, "", 0)
                                
            self.configured = True                        
            self.reset_srv = rospy.Service("~reset", Reset, self.reset_cb)
            self.step_srv = rospy.Service("~step", Step, self.step_cb)
            rospy.logwarn("[{}] configured!".format(self.name))
                    
            return ConfigureResponse(True, self.state_dim, self.actions_dim, self.servo_control, self.max_action_lim)
        else:
            rospy.logwarn("[{}] interface already has been congigured.".format(self.name, action_el))                                
            return ConfigureResponse(True, self.state_dim, self.actions_dim, self.servo_control, self.max_action_lim)
                                
    
    def step_cb(self, req):
        # clear fall data        
        self.last_episode_fall = []
        self.set_action(req.action)
        rospy.sleep(req.step_duration_sec)
        self.last_action = req.action
        state, reward = self.get_state(True)
        return StepResponse(state, reward, self.check_done())
    
    def reset_cb(self, req):
        self.set_action([0] * self.actions_dim)
        self.reset_sim_srv()        
        return ResetResponse(self.get_state())
    
    #
    # LOW INTERFACE FUNCTIONS
    #
    def unrm(self, val):        
        return unnorm(val, -self.max_action_lim, self.max_action_lim)                
    
    def set_action(self, action):
        index = 0
        for action_el in self.requested_actions:
            if action_el == "sync_legs_vel":
                self.foot_l_pub.publish(self.unrm(action[index]))                
                self.foot_r_pub.publish(self.unrm(action[index]))
                index+=1
                self.knee_l_pub.publish(self.unrm(action[index]))                
                self.knee_r_pub.publish(self.unrm(action[index]))
                index+=1
                self.hip_l_pub.publish(self.unrm(action[index]))                
                self.hip_r_pub.publish(self.unrm(action[index]))
                index+=1                
            elif action_el == "left_legs_vel":
                self.foot_l_pub.publish(self.unrm(action[index]))
                index+=1
                self.knee_l_pub.publish(self.unrm(action[index]))
                index+=1
                self.hip_l_pub.publish(self.unrm(action[index]))
                index+=1
            elif action_el == "right_legs_vel":
                self.foot_r_pub.publish(self.unrm(action[index]))
                index+=1
                self.knee_r_pub.publish(self.unrm(action[index]))
                index+=1
                self.hip_r_pub.publish(self.unrm(action[index]))
                index+=1
            elif action_el == "all_head_vel":
                self.neck_pub.publish(self.unrm(action[index]))
                index+=1
                self.head_pub.publish(self.unrm(action[index]))
                index+=1                                                           
    
    def check_done(self):
        return True in self.last_episode_fall        
    
    # ==============
    # rewards
    # rewards
    # rewards
    # ==============
    def stup_reward_z_1(self, ind_base):
        return -(0.26 - self.last_link_states.pose[ind_base].position.z)**2
    
    def stup_reward_z_2(self, ind_base):
        return -np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z)
    
    def stup_reward_z_3(self, ind_base):
        return 0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z)
    
    def stup_reward_z_mimimize_actions(self, ind_base):
        return 0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z) + (3 - np.sum(np.absolute(np.array(self.last_action))))
        
    def stup_reward_z_contacts_1(self, ind_base):
        contacts = 5 / (1+int(self.last_feet_contacts.foot_l) + int(self.last_feet_contacts.foot_r) + int(self.last_feet_contacts.heel_l) + int(self.last_feet_contacts.heel_r))
        return (0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z)) * contacts
    
    def stup_reward_z_contacts_2(self, ind_base):
        contacts = 4 / (int(self.last_feet_contacts.foot_l) + int(self.last_feet_contacts.foot_r) + int(self.last_feet_contacts.heel_l) + int(self.last_feet_contacts.heel_r))
        return (0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z)) + contacts
    
    def stup_reward_z_contacts_3(self, ind_base):
        contacts = int(self.last_feet_contacts.foot_l) + int(self.last_feet_contacts.foot_r) + int(self.last_feet_contacts.heel_l) + int(self.last_feet_contacts.heel_r)
        return (0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z)) * contacts
    
    def stup_reward_z_contacts_4(self, ind_base):
        contacts = int(self.last_feet_contacts.foot_l) + int(self.last_feet_contacts.foot_r) + int(self.last_feet_contacts.heel_l) + int(self.last_feet_contacts.heel_r)
        return (0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z)) + 0.01 * contacts
    
    def stup_reward_z_fall_penalty_1(self, ind_base):
        return -np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z) - int(self.check_done())
    
    def stup_reward_z_pitch_vel_1(self, ind_base):
        P = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        return -((0.26 - self.last_link_states.pose[ind_base].position.z)**2 + 0.1*(P)**2 + 0.01*(self.last_link_states.twist[ind_base].linear.x**2 + self.last_link_states.twist[ind_base].linear.y**2 + self.last_link_states.twist[ind_base].linear.z**2))
    
    def stup_reward_z_pitch_1(self, ind_base):        
        P = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        return 0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z) + 0.05 * (np.pi - np.absolute(P))       
    
    def stup_reward_z_pitch_2(self, ind_base):        
        P = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        return 0.3-np.absolute(0.3 - self.last_link_states.pose[ind_base].position.z) + 0.01 * (np.pi - np.absolute(P))       
    
    def stup_reward_z_com_cop_1(self, ind_base):
        z_part = (0.26 - self.last_link_states.pose[ind_base].position.z)                
        com_cop_part = (self.last_mass_center.x - self.last_press_center.x)**2 + (self.last_mass_center.y - self.last_press_center.y)**2
        return -z_part**2 - com_cop_part
        
    def stup_reward_z_pitch_com_cop_1(self, ind_base):
        z_part = (0.26 - self.last_link_states.pose[ind_base].position.z)                
        com_cop_part = (self.last_mass_center.x - self.last_press_center.x)**2 + (self.last_mass_center.y - self.last_press_center.y)**2
        P = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        return -z_part**2 - com_cop_part - 0.1 * P**2
        
    def stup_reward_z_body_pitch_com_cop_head_pitch_1(self, ind_base):
        z_part = (0.26 - self.last_link_states.pose[ind_base].position.z)                
        
        com_cop_part = (self.last_mass_center.x - self.last_press_center.x)**2 + (self.last_mass_center.y - self.last_press_center.y)**2
        
        bodyP = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        
        ind_head = self.last_link_states.name.index("bd1::head_link")
        quat = self.last_link_states.pose[ind_head].orientation
        headP = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[1]
        
        return -z_part**2 - com_cop_part - 0.1 * bodyP**2 - 0.1 * headP**2
    
    def stup_reward_z_body_pitch_com_cop_head_pitch_2(self, ind_base):
        z_part = (0.26 - self.last_link_states.pose[ind_base].position.z)                
        
        com_cop_part = (self.last_mass_center.x - self.last_press_center.x)**2 + (self.last_mass_center.y - self.last_press_center.y)**2
        
        bodyP = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        
        ind_head = self.last_link_states.name.index("bd1::head_link")
        quat = self.last_link_states.pose[ind_head].orientation
        headP = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[1]
        
        return -z_part**2 - 0.5 * com_cop_part - 0.1 * bodyP**2 - 0.1 * headP**2
    
    def stup_reward_z_body_pitch_com_cop_head_pitch_3(self, ind_base):
        z_part = (0.26 - self.last_link_states.pose[ind_base].position.z)                
        
        com_cop_part = (self.last_mass_center.x - self.last_press_center.x)**2 + (self.last_mass_center.y - self.last_press_center.y)**2
        
        bodyP = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        
        ind_head = self.last_link_states.name.index("bd1::head_link")
        quat = self.last_link_states.pose[ind_head].orientation
        headP = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[1]                
        
        return 10*-z_part**2 - com_cop_part - 0.1 * bodyP**2 - 0.1 * headP**2 - int(self.check_done())        
    
    def just_fall_reward(self, ind_base):
        return int(self.check_done())
    
    def just_Z_reward(self, ind_base):
        return self.last_link_states.pose[ind_base].position.z
    
    def max_z_and_body_pitch_1(self, ind_base):
        P = euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1]
        return self.last_link_states.pose[ind_base].position.z + (1 - np.absolute(np.sin(P)))
    
    def walk_reward_max_vx(self, ind_base):
        return self.last_link_states.twist[ind_base].linear.x
    
    # STATE
    
    def get_state(self, get_reward = False):
        
        state = []
        
        # Model State                
        #model_state = self.get_model_state_srv("bd1","")
        
        ind_base = self.last_link_states.name.index("bd1::base_link")
        for state_el in self.requested_state:
            ## position as if            
            
            if state_el == "base_pose":                
                state.append(self.last_link_states.pose[ind_base].position.x)
                state.append(self.last_link_states.pose[ind_base].position.y)
                state.append(self.last_link_states.pose[ind_base].position.z)
                #state.append(model_state.pose.position.x)
                #state.append(model_state.pose.position.y)
                #state.append(model_state.pose.position.z)
                
            #elif state_el == "base_rot_rpy":        
                #rpy = euler_from_quaternion([model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w])                
                #state += rpy
                
            elif state_el == "base_rot_quat":
                #state.append(model_state.pose.orientation.x)
                #state.append(model_state.pose.orientation.y)
                #state.append(model_state.pose.orientation.z)
                #state.append(model_state.pose.orientation.w)            
                state.append(self.last_link_states.pose[ind_base].orientation.x)
                state.append(self.last_link_states.pose[ind_base].orientation.y)
                state.append(self.last_link_states.pose[ind_base].orientation.z)
                state.append(self.last_link_states.pose[ind_base].orientation.w)
            # linear velocities as if
            elif state_el == "base_twist_lin":                
                state.append(self.last_link_states.twist[ind_base].linear.x)
                state.append(self.last_link_states.twist[ind_base].linear.y)
                state.append(self.last_link_states.twist[ind_base].linear.z)                
            elif state_el == "base_twist_ang":
                state.append(self.last_link_states.twist[ind_base].angular.x)
                state.append(self.last_link_states.twist[ind_base].angular.y)
                state.append(self.last_link_states.twist[ind_base].angular.z)
            # Joints Positions as if
            ## Joint State topic is bad and laggy
            #elif state_el == "left_leg_pos":
                #state.append(self.last_joint_states.position[0])
                #state.append(self.last_joint_states.position[3])
                #state.append(self.last_joint_states.position[6])
            ## Joints Positions normalized
            #elif state_el == "left_leg_pos_norm":
                #state.append( np.sin(self.last_joint_states.position[0] ))
                #state.append( np.cos(self.last_joint_states.position[0] ))
                #state.append( np.sin(self.last_joint_states.position[3] ))
                #state.append( np.cos(self.last_joint_states.position[3] ))
                #state.append( np.sin(self.last_joint_states.position[6] ))
                #state.append( np.cos(self.last_joint_states.position[6] ))
            #elif state_el == "left_leg_vel":
                #state.append(self.last_joint_states.velocity[0])
                #state.append(self.last_joint_states.velocity[3])
                #state.append(self.last_joint_states.velocity[6])
            ## Joints Positions as if
            #elif state_el == "right_leg_pos":
                #state.append(self.last_joint_states.position[1])
                #state.append(self.last_joint_states.position[4])
                #state.append(self.last_joint_states.position[7])
            ## Joints Positions normalized
            #elif state_el == "right_leg_pos_norm":
                #state.append( np.sin(self.last_joint_states.position[1] ))
                #state.append( np.cos(self.last_joint_states.position[1] ))
                #state.append( np.sin(self.last_joint_states.position[4] ))
                #state.append( np.cos(self.last_joint_states.position[4] ))
                #state.append( np.sin(self.last_joint_states.position[7] ))
                #state.append( np.cos(self.last_joint_states.position[7] ))
            #elif state_el == "right_leg_vel":
                #state.append(self.last_joint_states.velocity[1])
                #state.append(self.last_joint_states.velocity[4])
                #state.append(self.last_joint_states.velocity[7])
            #elif state_el == "all_head_pos":
                #state.append(self.last_joint_states.position[2])
                #state.append(self.last_joint_states.position[5])
            #elif state_el == "all_head_pos_norm":
                #state.append( np.sin(self.last_joint_states.position[2] ))
                #state.append( np.cos(self.last_joint_states.position[2] ))
                #state.append( np.sin(self.last_joint_states.position[5] ))
                #state.append( np.cos(self.last_joint_states.position[5] ))
            #elif state_el == "all_head_vel":
                #state.append(self.last_joint_states.velocity[2])
                #state.append(self.last_joint_states.velocity[5])     
            elif state_el == "com_abs":
                state.append(self.last_mass_center.x)
                state.append(self.last_mass_center.y)
                state.append(self.last_mass_center.z)
            elif state_el == "cop_abs":
                state.append(self.last_press_center.x)
                state.append(self.last_press_center.y)
                state.append(self.last_press_center.z)
            elif state_el == "head_rot_quat":
                ind_head = self.last_link_states.name.index("bd1::head_link")
                quat = self.last_link_states.pose[ind_head].orientation
                state.append(quat.x)
                state.append(quat.y)
                state.append(quat.z)
                state.append(quat.w)
            elif state_el == "left_leg_all_quats":
                for link in ["bd1::hip_l_link", "bd1::knee_l_link","bd1::foot_l_link"]:
                    ind = self.last_link_states.name.index(link)
                    quat = self.last_link_states.pose[ind].orientation
                    state.append(quat.x)
                    state.append(quat.y)
                    state.append(quat.z)
                    state.append(quat.w)
            elif state_el == "right_leg_all_quats":
                for link in ["bd1::hip_r_link", "bd1::knee_r_link","bd1::foot_r_link"]:
                    ind = self.last_link_states.name.index(link)
                    quat = self.last_link_states.pose[ind].orientation
                    state.append(quat.x)
                    state.append(quat.y)
                    state.append(quat.z)
                    state.append(quat.w)
            elif state_el == "feet_contacts":
                state.append(float(self.last_feet_contacts.foot_r))
                state.append(float(self.last_feet_contacts.foot_l))
                state.append(float(self.last_feet_contacts.heel_r))
                state.append(float(self.last_feet_contacts.heel_l))
                
        
        if get_reward:            
            return state, self.requested_reward(ind_base) 
        else:
            return state               
        
    #
    # DATA COLLECTING
    #
    def joint_states_cb(self, msg):
        self.last_joint_states = msg        
        
    def fall_cb(self, msg):
        self.last_episode_fall.append(msg.data)
        
    def com_cb(self, msg):
        self.last_mass_center = msg.point
        
    def cop_cb(self, msg):
        self.last_press_center = msg.point
        
    def link_states_cb(self, msg):
        self.last_link_states = msg
        
    def feet_contacts_cb(self, msg):
        self.last_feet_contacts = msg;
        
    def run(self):        
        rospy.spin()
        
if __name__ == '__main__' :
    ugei = UniversalGazeboEnvironmentInterface()
    ugei.run()

