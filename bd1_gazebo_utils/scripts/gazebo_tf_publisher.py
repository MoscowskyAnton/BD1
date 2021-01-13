#!/usr/bin/env python
# coding: utf-8

import rospy
from gazebo_msgs.msg import LinkStates
import tf2_ros
from geometry_msgs.msg import TransformStamped

class GazeboTfBroadcaster(object):
    
    def __init__(self):
        rospy.init_node('gazebo_tf_broadcaster')
        self.name = rospy.get_name()
        
        self.br = tf2_ros.TransformBroadcaster()
        
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_cb)
        
    def link_states_cb(self, msg):
        try:
            ind = msg.name.index('bd1::base_link')
            msg.pose[ind]
            
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "map"
            t.child_frame_id = 'base_link'
            t.transform.rotation = msg.pose[ind].orientation
            t.transform.translation.x = msg.pose[ind].position.x
            t.transform.translation.y = msg.pose[ind].position.y
            t.transform.translation.z = msg.pose[ind].position.z
            
            self.br.sendTransform(t)
            
        except ValueError:
            rospy.logwarn("[{}] not found bd1::base_link in /gazebo/link_states".format(self.name))
            return            
            
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    gtfb = GazeboTfBroadcaster()
    gtfb.run()
    
