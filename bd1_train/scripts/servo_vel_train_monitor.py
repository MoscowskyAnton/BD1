#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import Bool, Float64
import matplotlib.pyplot as plt
from collections import deque

class ServoTrainMonitor(object):
    def __init__(self):
        rospy.init_node('servo_train_monitor')
        
        self.print_data = [[], [] , [] , [] , [], [], [], []]
        self.fall = False        
        
        self.episode_store = 5      
        self.cntr = 0
        #self.up_r_servo = deque(maxlen = 10)
        self.print_it = False
        rospy.Subscriber('head_servo_velocity_controller/command', Float64, self.head_cb)
        rospy.Subscriber('neck_servo_velocity_controller/command', Float64, self.neck_cb)
        
        rospy.Subscriber('leg_up_r_servo_velocity_controller/command', Float64, self.up_r_cb)
        rospy.Subscriber('leg_mid_r_servo_velocity_controller/command', Float64, self.mid_r_cb)
        rospy.Subscriber('feet_r_servo_velocity_controller/command', Float64, self.feet_r_cb)
        
        rospy.Subscriber('leg_up_l_servo_velocity_controller/command', Float64, self.up_l_cb)
        rospy.Subscriber('leg_mid_l_servo_velocity_controller/command', Float64, self.mid_l_cb)
        rospy.Subscriber('feet_l_servo_velocity_controller/command', Float64, self.feet_l_cb)
        
        rospy.Subscriber("fall_detector/fall", Bool, self.fall_cb)
                
    
    def head_cb(self, msg):
        if not self.fall:
            self.print_data[0].append(msg.data)
    
    def neck_cb(self, msg):
        if not self.fall:
            self.print_data[1].append(msg.data)
        
    def up_r_cb(self, msg):
        if not self.fall:
            self.print_data[2].append(msg.data)
        
    def mid_r_cb(self, msg):
        if not self.fall:
            self.print_data[3].append(msg.data)
        
    def feet_r_cb(self, msg):
        if not self.fall:
            self.print_data[4].append(msg.data)
        
    def up_l_cb(self, msg):
        if not self.fall:
            self.print_data[5].append(msg.data)
        
    def mid_l_cb(self, msg):
        if not self.fall:
            self.print_data[6].append(msg.data)
        
    def feet_l_cb(self, msg):
        if not self.fall:
            self.print_data[7].append(msg.data)
    
    def fall_cb(self, msg):
        #self.print_data[8].append(msg.data)
        if self.fall and not msg.data:
            self.print_it = True
        self.fall = msg.data        
        
    
    def run(self):
        while(not rospy.is_shutdown()):                        
            
            if self.print_it:
                plt.plot(self.print_data[2],'-r', alpha = 0.5)
                plt.plot(self.print_data[3],'-b', alpha = 0.5)
                plt.plot(self.print_data[4],'-g', alpha = 0.5)
                plt.pause(0.05)           
                self.print_it = False
                self.cntr +=1
                if self.cntr > self.episode_store:
                    self.print_data = [[], [] , [] , [] , [], [], [], []]
                    plt.cla()
                                

if __name__ == '__main__' :
    stm = ServoTrainMonitor()
    stm.run()
    

    
        
    
    
    
