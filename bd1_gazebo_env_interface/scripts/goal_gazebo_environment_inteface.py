#!/usr/bin/env python
# coding: utf-8

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, SetModelState, SetLinkState, GetLinkState, SetModelConfiguration
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
import numpy as np
from bd1_gazebo_env_interface.srv import GoalEnvStep, GoalEnvReset, GoalEnvStepResponse, GoalEnvResetResponse, GoalEnvConfigure, GoalEnvConfigureResponse
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import LinkStates
from bd1_gazebo_utils.msg import FeetContacts
from gazebo_msgs.msg import ModelState, LinkState
from bd1_gazebo_utils.srv import SetState

def unnorm(x, x_min, x_max):    
        return ((x+1)/2)*(x_max-x_min)  + x_min

class GoalGazeboEnvironmentInterface(object):
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
        rospy.loginfo("[{}] waiting for services...".format(self.name))
        rospy.wait_for_service('gazebo/reset_simulation')
        self.reset_sim_srv = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        rospy.wait_for_service('gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        
        rospy.wait_for_service('gazebo/set_link_state')
        self.set_link_state_srv = rospy.ServiceProxy('gazebo/set_link_state', SetLinkState)
        
        rospy.wait_for_service('gazebo/get_link_state')
        self.get_link_state_srv = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        
        rospy.wait_for_service('gazebo/unpause_physics')
        self.unpause_srv = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        
        rospy.wait_for_service('gazebo/pause_physics')
        self.pause_srv = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        
        #rospy.wait_for_service('gazebo_state_recorder/set_state')
        #self.set_robot_state_srv = rospy.ServiceProxy('gazebo_state_recorder/set_state', SetState)
        
        rospy.wait_for_service('gazebo/set_model_configuration')
        self.set_model_config_srv = rospy.ServiceProxy('gazebo/set_model_configuration', SetModelConfiguration)
        rospy.loginfo("[{}] all services are ready!".format(self.name))
        
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
        
        start_state = rospy.get_param("~start_state",'SIT')
        # ['neck_j', 'head_j', 'hip_l_j', 'hip_r_j', 'knee_r_j', 'knee_l_j', 'foot_l_j', 'foot_r_j']
        if start_state == 'SIT':
            self.start_state = [1.5,1.5,-1.5,-1.5,3,3,1.5,1.5]
        elif start_state == 'DEPLOY': 
            self.start_state = [0.75,0.75,-0.75,-0.75,1.,1.,0.0,0.0]
        elif start_state == 'STAND':
            self.start_state = [0.,0.,0.,0.,0.,0.,0.0,0.0]
        elif start_state == 'DEPLOY2':
            self.start_state = [0.75,0.75,-0.25,-0.25,0.5,0.5,0.25,0.25]
        else:
            rospy.logerr("[{}] can't find start state {}! Exit.".format(self.name, start_state))
            return
        # # sit
        
        self.init_link_states = [["bd1::base_link", "bd1::neck_link", -1.5],
                                 ["bd1::neck_link", "bd1::head_link", 1.5],
                                 ["bd1::base_link", "bd1::hip_r_link", 0],
                                 ["bd1::base_link", "bd1::hip_l_link", 0],
                                 ["bd1::hip_r_link", "bd1::knee_r_link", -3],
                                 ["bd1::hip_l_link", "bd1::knee_l_link", -3],
                                 ["bd1::knee_r_link", "bd1::foot_r_link", 0],
                                 ["bd1::knee_l_link", "bd1::foot_l_link", 0]]
        
        self.state_types = {"base_pose": 3,
                            "base_z": 1,
                            "base_rot_quat":4,
                            "base_rot_r":1,
                            "base_rot_p":1,
                            "base_rot_y":1,
                            "base_twist_lin":3,
                            "base_twist_ang":3,
                            "com_abs":3,
                            "cop_abs":3,
                            "com_rel":3,
                            "cop_rel":3,
                            "head_rot_quat":4,
                            "neck_rot_quat":4,
                            "left_leg_all_quats":12,
                            "right_leg_all_quats":12,
                            "feet_contacts":4,
                            "all_servo_joints":8,
                            "all_servo_vels":8}
        
        self.state_highs = {"base_pose": [1, 1, 1],
                            "base_z": [1],
                            "base_rot_quat":[1, 1, 1, 1],                            
                            "base_rot_r":[np.pi],
                            "base_rot_p":[np.pi],
                            "base_rot_y":[np.pi],
                            "base_twist_lin":[np.inf, np.inf, np.inf],
                            "base_twist_ang":[np.inf, np.inf, np.inf],
                            "com_abs":[1,1,1],
                            "cop_abs":[1,1,1],
                            "com_rel":[1,1,1],
                            "cop_rel":[1,1,1],
                            "head_rot_quat":[1,1,1,1],
                            "neck_rot_quat":[1,1,1,1],
                            "left_leg_all_quats":[1]*12,
                            "right_leg_all_quats":[1]*12,
                            "feet_contacts":[1,1,1,1],
                            "all_servo_joints":[np.pi/2, np.pi/2, np.pi/2, 0, 0, np.pi, np.pi, np.pi/2],
                            "all_servo_vels":[self.max_action_lim] * 8}
        
        self.state_lows = {"base_pose": [-1, -1, 0],
                           "base_z": [0],
                            "base_rot_quat":[-1, -1, -1, -1],
                            "base_rot_r":[-np.pi],
                            "base_rot_p":[-np.pi],
                            "base_rot_y":[-np.pi],
                            "base_twist_lin":[-np.inf, -np.inf, -np.inf],
                            "base_twist_ang":[-np.inf, -np.inf, -np.inf],
                            "com_abs":[-1,-1,0],
                            "cop_abs":[-1,-1,0],
                            "com_rel":[-1,-1,0],
                            "cop_rel":[-1,-1,0],
                            "head_rot_quat":[-1,-1,-1,-1],
                            "neck_rot_quat":[-1,-1,-1,-1],
                            "left_leg_all_quats":[-1]*12,
                            "right_leg_all_quats":[-1]*12,
                            "feet_contacts":[0,0,0,0],
                            "all_servo_joints":[-np.pi/2, -np.pi/2, 0, -np.pi/2, -np.pi/2, 0, 0, 0],
                            "all_servo_vels":[-self.max_action_lim] * 8}
        
        self.state_high = []
        self.state_low = []
        
        self.actions_types = {"sync_legs_vel":3,
                             "left_legs_vel":3,
                             "right_legs_vel":3,
                             "all_head_vel":2}
        
        
        self.requested_state = []
        self.requested_actions = []
        self.requested_goal = []
        self.requested_reward = None
        self.configured = False
        self.state_dim = 0
        self.actions_dim = 0
        self.config_srv = rospy.Service("~configure", GoalEnvConfigure, self.config_cb)
        self.reset_srv = rospy.Service("~reset", GoalEnvReset, self.reset_cb)
        rospy.logwarn("[{}] awaiting configuration...".format(self.name))                        
        
    
    #
    # HIGH INTERFACE FUNCTIONS
    #
    def config_cb(self, req):
        if not self.configured:
            self.state_dim = 0
            self.actions_dim = 0
            self.state_high = []
            self.state_low = []
            
            self.goal_high = []
            self.goal_low = []
            
            self.goal_dim = 0
            #rospy.logwarn("[{}] got config.".format(self.name))
            
            for state_el in req.state:
                if state_el in self.state_types:
                    self.state_dim += self.state_types[state_el]                 
                    self.requested_state.append(state_el)
                    self.state_high += self.state_highs[state_el]
                    self.state_low += self.state_lows[state_el]
                else:
                    rospy.logerr("[{}] unknown state element {} try to config again!".format(self.name, state_el))                
                    return GoalEnvConfigureResponse(False, 0, 0, 0, "", 0, [], [], [], [])
                
            for goal_el in req.goal:
                if goal_el in self.state_types:
                    self.goal_dim += self.state_types[goal_el]                 
                    self.requested_goal.append(goal_el)
                    self.goal_high += self.state_highs[goal_el]
                    self.goal_low += self.state_lows[goal_el]
                else:
                    rospy.logerr("[{}] unknown state element {} try to config again!".format(self.name, goal_el))                
                    return GoalEnvConfigureResponse(False, 0, 0, 0, "", 0, [], [], [], [])
                                    
            for action_el in req.actions:
                if action_el in self.actions_types:
                    self.actions_dim += self.actions_types[action_el]
                    self.requested_actions.append(action_el)
                else:
                    rospy.logerr("[{}] unknown action element {} try to config again!".format(self.name, action_el))                                          
                    return GoalEnvConfigureResponse(False, 0, 0, 0, "", 0, [], [], [], [])
            
            if self.state_dim == 0:
                rospy.logerr("[{}] state vector is zero! Interface isn't configured, try again.".format(self.name))                                
                return GoalEnvConfigureResponse(False, 0, 0, 0, "", 0, [], [], [], [])
                
            if self.actions_dim == 0:
                rospy.logerr("[{}] action vector is zero! Interface isn't configured, try again.".format(self.name))                                
                return GoalEnvConfigureResponse(False, 0, 0, 0, "", 0, [], [], [], [])
            
                                
            self.configured = True                                    
            self.step_srv = rospy.Service("~step", GoalEnvStep, self.step_cb)
            rospy.logwarn("[{}] configured!".format(self.name))
        
                    
            return GoalEnvConfigureResponse(True, self.state_dim, self.actions_dim, self.goal_dim, self.servo_control, self.max_action_lim, self.state_high, self.state_low, self.goal_high, self.goal_low)
        else:
            rospy.logwarn("[{}] interface already has been configured.".format(self.name))                                
            return GoalEnvConfigureResponse(True, self.state_dim, self.actions_dim, self.goal_dim, self.servo_control, self.max_action_lim, self.state_high, self.state_low, self.goal_high, self.goal_low)
                                
    
    def step_cb(self, req):
        # clear fall data        
        self.last_episode_fall = []
        self.set_action(req.action)
        rospy.sleep(req.step_duration_sec)
        self.last_action = req.action
        state = self.get_state(self.requested_state)
        ach_goal = self.get_state(self.requested_goal)
        return GoalEnvStepResponse(state, ach_goal, self.desired_goal, self.check_done())
    
    def reset_cb(self, req):
        if self.configured:
            self.set_action([0] * self.actions_dim)
            
            
        self.set_model_config_srv('bd1', 'robot_description',['neck_j', 'head_j', 'hip_l_j', 'hip_r_j', 'knee_r_j', 'knee_l_j', 'foot_l_j', 'foot_r_j'],self.start_state)

        state = ModelState()
        state.pose.position.z = 0.1 + 0.07

        state.model_name = "bd1"
        self.set_model_state_srv(state)
        
        self.random_goal()

        if self.configured:
            return GoalEnvResetResponse(self.get_state(self.requested_state),
                                 self.get_state(self.requested_goal),
                                 self.desired_goal)
    
    #
    # LOW INTERFACE FUNCTIONS
    #
    def random_goal(self):
        self.desired_goal = np.random.uniform(self.goal_low, self.goal_high)
        rospy.logwarn(f"[{self.name}] new goal {self.desired_goal}")
    
    def full_reset(self):
        for l_st_t in self.init_link_states:
            #g_l_st = self.get_link_state_srv(l_st_t[1], l_st_t[0]).link_state            
            l_st = LinkState()
            l_st.link_name = l_st_t[1]
            l_st.reference_frame = l_st_t[0]
            #l_st.pose.position = g_l_st.pose.position
            quat = quaternion_from_euler(0, l_st_t[2], 0)
            l_st.pose.orientation.x = quat[0]
            l_st.pose.orientation.y = quat[1]
            l_st.pose.orientation.z = quat[2]
            l_st.pose.orientation.w = quat[3]            
            self.set_link_state_srv(l_st)
        
    
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
    
    # STATE
    
    def get_state(self, requested_state):
        
        state = []        
        
        # Model State                
        #model_state = self.get_model_state_srv("bd1","")
        
        ind_base = self.last_link_states.name.index("bd1::base_link")
        for state_el in requested_state:
            ## position as if            
            
            if state_el == "base_pose":                
                state.append(self.last_link_states.pose[ind_base].position.x)
                state.append(self.last_link_states.pose[ind_base].position.y)
                state.append(self.last_link_states.pose[ind_base].position.z)
            elif state_el == "base_z":
                state.append(self.last_link_states.pose[ind_base].position.z)
            elif state_el == "base_rot_quat":                            
                state.append(self.last_link_states.pose[ind_base].orientation.x)
                state.append(self.last_link_states.pose[ind_base].orientation.y)
                state.append(self.last_link_states.pose[ind_base].orientation.z)
                state.append(self.last_link_states.pose[ind_base].orientation.w)
            elif state_el == "base_rot_r":
                state.append(euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[0])                
            elif state_el == "base_rot_p":
                state.append(euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[1])
            elif state_el == "base_rot_y":
                state.append(euler_from_quaternion([self.last_link_states.pose[ind_base].orientation.x, self.last_link_states.pose[ind_base].orientation.y, self.last_link_states.pose[ind_base].orientation.z, self.last_link_states.pose[ind_base].orientation.w])[2])
            # linear velocities as if
            elif state_el == "base_twist_lin":                
                state.append(self.last_link_states.twist[ind_base].linear.x)
                state.append(self.last_link_states.twist[ind_base].linear.y)
                state.append(self.last_link_states.twist[ind_base].linear.z)                
            elif state_el == "base_twist_ang":
                state.append(self.last_link_states.twist[ind_base].angular.x)
                state.append(self.last_link_states.twist[ind_base].angular.y)
                state.append(self.last_link_states.twist[ind_base].angular.z)    
            elif state_el == "com_abs":
                state.append(self.last_mass_center.x)
                state.append(self.last_mass_center.y)
                state.append(self.last_mass_center.z)
            elif state_el == "cop_abs":
                state.append(self.last_press_center.x)
                state.append(self.last_press_center.y)
                state.append(self.last_press_center.z)
            elif state_el == "com_rel":
                state.append(self.last_mass_center.x - self.last_link_states.pose[ind_base].position.x)
                state.append(self.last_mass_center.y - self.last_link_states.pose[ind_base].position.y)
                state.append(self.last_mass_center.z - self.last_link_states.pose[ind_base].position.z)
            elif state_el == "cop_rel":
                state.append(self.last_press_center.x - self.last_link_states.pose[ind_base].position.x)
                state.append(self.last_press_center.y - self.last_link_states.pose[ind_base].position.y)
                state.append(self.last_press_center.z - self.last_link_states.pose[ind_base].position.z)
            elif state_el == "head_rot_quat":
                ind_head = self.last_link_states.name.index("bd1::head_link")
                quat = self.last_link_states.pose[ind_head].orientation
                state.append(quat.x)
                state.append(quat.y)
                state.append(quat.z)
                state.append(quat.w)
            elif state_el == "neck_rot_quat":
                ind_head = self.last_link_states.name.index("bd1::neck_link")
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
            elif state_el == "all_servo_joints":
                state += self.last_joint_states.position
            elif state_el == "all_servo_vels":
                state += self.last_joint_states.velocity    
        
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
    ggei = GoalGazeboEnvironmentInterface()
    ggei.run()


