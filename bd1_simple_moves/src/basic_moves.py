#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import Float64
from bd1_simple_moves.srv import SetLegs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Empty
import numpy as np

class BasicMoves(object):
    def __init__(self):
        rospy.init_node('basic_moves')            
        
        self.left_leg_pub = rospy.Publisher("/left_leg_servo_states_controller/command", JointTrajectory, queue_size = 10)
        
        self.right_leg_pub = rospy.Publisher("/right_leg_servo_states_controller/command", JointTrajectory, queue_size = 10)
        
        self.head_pub = rospy.Publisher("/head_servo_state_controller/command", JointTrajectory, queue_size = 10)
        
        self.stop_time = None        
        
        rospy.Service('~set_legs', SetLegs, self.set_legs_cb)
        
        rospy.Service('~deploy', Empty, self.deploy_cb)
        
        rospy.Service('~conseal', Empty, self.conseal_cb)
        
        rospy.Service('~strech', Empty, self.strech_cb)
        
    
    def deploy_cb(self, req):
        self.send_leg_cmd(0.5, 0.5, -1, -1, 0.5, 0.5, 1)    
        self.send_head_cmd(-0.5, 0.5, 1)    
        return []
    
    def strech_cb(self, req):
        self.send_leg_cmd(0, 0, 0, 0, 0, 0, 1)    
        self.send_head_cmd(0, 0, 1)    
        return []
    
    def conseal_cb(self, req):
        self.send_leg_cmd(1.5, 1.5, -3, -3, 1.5, 1.5, 1)        
        #self.send_head_cmd(-np.pi/2, np.pi/2, 1)
        self.send_head_cmd(-1.5, 1.5, 1)    
        return []
    
    def send_head_cmd(self, neck, head, velocity):
        
        head_cmd = JointTrajectory()
        head_cmd.header.stamp = rospy.Time.now()
        
        head_cmd.joint_names = ['neck_j', 'head_j']
        p = JointTrajectoryPoint()
        p.positions.append(neck)
        p.positions.append(head)
        
        p.velocities.append(velocity)
        p.velocities.append(velocity)        
        
        p.time_from_start = rospy.Duration(1.0);
        head_cmd.points.append(p)        
                
                
        self.head_pub.publish(head_cmd)
    
    def set_legs_cb(self, req):
        self.send_leg_cmd(req.up_l, req.up_r, req.mid_l, req.mid_r, req.feet_l, req.feet_r, req.speed)        
        return []
    
    def send_leg_cmd(self, up_l, up_r, mid_l, mid_r, feet_l, feet_r, velocity):

        left_leg = JointTrajectory()
        left_leg.header.stamp = rospy.Time.now()
        
        left_leg.joint_names = ['up_leg_l_j', 'mid_leg_l_j', 'feet_l_j']
        p = JointTrajectoryPoint()
        p.positions.append(up_l)
        p.positions.append(mid_l)
        p.positions.append(feet_l)
        p.velocities.append(velocity)
        p.velocities.append(velocity*2)        
        p.velocities.append(velocity)        
        p.time_from_start = rospy.Duration(1.0);
        left_leg.points.append(p)        
        
        #print(left_leg)
        
        right_leg = JointTrajectory()
        right_leg.header.stamp = rospy.Time.now()
        
        right_leg.joint_names = ['up_leg_r_j', 'mid_leg_r_j', 'feet_r_j']
        p = JointTrajectoryPoint()
        p.positions.append(up_r)
        p.velocities.append(velocity)      
        p.positions.append(mid_r)
        p.velocities.append(velocity*2)      
        p.positions.append(feet_r)
        p.velocities.append(velocity)            
        p.time_from_start = rospy.Duration(1.0);
        right_leg.points.append(p)        
        
        #print(right_leg)
        
        self.left_leg_pub.publish(left_leg)
        self.right_leg_pub.publish(right_leg)        
    
    def run(self):
        rospy.spin()
    
    
    
if __name__ == '__main__' :
    bm = BasicMoves()
    bm.run()
    
