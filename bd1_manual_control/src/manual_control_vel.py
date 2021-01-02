#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from bd1_manual_control.cfg import ManualControlVelConfig
from std_msgs.msg import Float64

class ManControl(object):
    def __init__(self):
        rospy.init_node("manual_control_vel")                
        
        # publishers
        self.head_pub = rospy.Publisher('head_servo_velocity_controller/command', Float64, queue_size = 1)
            
        self.neck_pub = rospy.Publisher('neck_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.hip_r_pub = rospy.Publisher('hip_r_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.knee_r_pub = rospy.Publisher('knee_r_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.foot_r_pub = rospy.Publisher('foot_r_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.hip_l_pub = rospy.Publisher('hip_l_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.knee_l_pub = rospy.Publisher('knee_l_servo_velocity_controller/command', Float64, queue_size = 1)
        
        self.foot_l_pub = rospy.Publisher('foot_l_servo_velocity_controller/command', Float64, queue_size = 1)                
        
        self.srv = Server(ManualControlVelConfig, self.callback)
        
    def callback(self, config, level):
        
        self.head_pub.publish(config["head_vel"])
        self.neck_pub.publish(config["neck_vel"])
        
        self.foot_l_pub.publish(config["foot_l_vel"])
        self.knee_l_pub.publish(config["knee_l_vel"])
        self.hip_l_pub.publish(config["hip_l_vel"])        
        
        if config["sync_legs"]:
            self.foot_r_pub.publish(config["foot_l_vel"])
            self.knee_r_pub.publish(config["knee_l_vel"])
            self.hip_r_pub.publish(config["hip_l_vel"])        
        else:
            self.foot_r_pub.publish(config["foot_r_vel"])
            self.knee_r_pub.publish(config["knee_r_vel"])
            self.hip_r_pub.publish(config["hip_r_vel"])        
                            
        return config
    
    def run(self):
        rospy.spin()
        
if __name__ == "__main__":
    mc = ManControl()
    mc.run()
        
       
     
    
