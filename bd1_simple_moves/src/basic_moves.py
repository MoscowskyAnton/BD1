#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import Float64
from bd1_simple_moves.srv import SetLegs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Empty

class TestServos(object):
    def __init__(self):
        rospy.init_node('test_servos')            
        
        self.left_leg_pub = rospy.Publisher("/left_leg_servo_states_controller/command", JointTrajectory, queue_size = 10)
        
        self.right_leg_pub = rospy.Publisher("/right_leg_servo_states_controller/command", JointTrajectory, queue_size = 10)
        
        self.stop_time = None        
        
        rospy.Service('~set_legs', SetLegs, self.set_legs_cb)
        
        rospy.Service('~deploy', Empty, self.deploy_cb)
        
        rospy.Service('~conseal', Empty, self.conseal_cb)
        
    def deploy_cb(self, req):
        self.send_cmd(0, 0, 0, 0, 0, 0, 1)        
        return []
    
    def conseal_cb(self, req):
        self.send_cmd(1.5, 1.5, -3, -3, 1.5, 1.5, 1)        
        return []
    
    def set_legs_cb(self, req):
        self.send_cmd(req.up_l, req.up_r, req.mid_l, req.mid_r, req.feet_l, req.feet_r, req.speed)        
        return []
    
    def send_cmd(self, up_l, up_r, mid_l, mid_r, feet_l, feet_r, velocity):

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
        
        return[]
    
    def run(self):
        rospy.spin()
    
    
    
if __name__ == '__main__' :
    ts = TestServos()
    ts.run()
    
