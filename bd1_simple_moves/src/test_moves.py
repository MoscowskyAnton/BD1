#!/usr/bin/env python
# coding: utf-8

import rospy
from std_msgs.msg import Float64
from bd1_simple_moves.srv import SetLegs

class TestServos(object):
    def __init__(self):
        rospy.init_node('test_servos')    
        
        self.up_r_pub = rospy.Publisher("/up_leg_right_state_controller/command",Float64, queue_size = 10)
        self.up_l_pub = rospy.Publisher("/up_leg_left_state_controller/command",Float64, queue_size = 10)
        
        self.mid_r_pub = rospy.Publisher("/mid_leg_right_state_controller/command",Float64, queue_size = 10)
        self.mid_l_pub = rospy.Publisher("/mid_leg_left_state_controller/command",Float64, queue_size = 10)
        
        self.feet_r_pub = rospy.Publisher("/feet_right_state_controller/command",Float64, queue_size = 10)
        self.feet_l_pub = rospy.Publisher("/feet_left_state_controller/command",Float64, queue_size = 10)
        
        self.stop_time = None        
        
        rospy.Service('~set_legs', SetLegs, self.set_legs_cb)
        rospy.Timer(rospy.Duration(0.01), self.timer_cb)
    
    def timer_cb(self, event):
        if( self.stop_time is not None ):
            if( self.stop_time < rospy.Time.now() ):
                self.send_cmd(0, 0, 0, 0, 0 ,0)
                self.stop_time = None
                
    def set_legs_cb(self, req):
        self.send_cmd(req.up_l, req.up_r, req.mid_l, req.mid_r, req.feet_l, req.feet_r)
        self.stop_time = rospy.Time.now() + rospy.Duration(req.duration)
        return []
    
    def send_cmd(self, up_l, up_r, mid_l, mid_r, feet_l, feet_r):
        self.up_r_pub.publish(up_r)
        self.up_l_pub.publish(up_l)
        self.mid_r_pub.publish(mid_r)
        self.mid_l_pub.publish(mid_l)
        self.feet_l_pub.publish(feet_l)
        self.feet_r_pub.publish(feet_r)
    
    def run(self):
        rospy.spin()
    
    
    
if __name__ == '__main__' :
    ts = TestServos()
    ts.run()
    
