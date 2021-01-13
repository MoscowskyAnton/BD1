#!/usr/bin/env python
# coding: utf-8
import yaml
import rospy
import pickle
from gazebo_msgs.msg import ModelState, LinkState
from gazebo_msgs.srv import GetModelState, GetLinkState, SetLinkState, SetModelState
from bd1_gazebo_utils.srv import RecordState, RecordStateResponse, SetState
from bd1_gazebo_utils import state_2_dict

class GazeboStateRecorder(object):
    
    def __init__(self):
        rospy.init_node('gazebo_state_recorder')
        
        self.name = rospy.get_name()
        
        self.save_path = rospy.get_param("~save_path", '/tmp')
        
        self.robot_state = [['bd1::base_link', 'bd1::hip_r_link'],
                            ['bd1::base_link', 'bd1::hip_l_link'],
                            ['bd1::base_link', 'bd1::hip_r_link'],
                            ['bd1::hip_l_link', 'bd1::knee_l_link'],
                            ['bd1::hip_r_link', 'bd1::knee_r_link'],
                            ['bd1::knee_l_link', 'bd1::foot_l_link'],
                            ['bd1::knee_l_link', 'bd1::foot_l_link'],
                            ['bd1::base_link', 'bd1::neck_link'],
                            ['bd1::neck_link', 'bd1::head_link']]
        
        rospy.wait_for_service('gazebo/get_model_state')
        self.get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        
        rospy.wait_for_service('gazebo/set_model_state')
        self.set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('gazebo/get_link_state')
        self.get_link_state_srv = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        
        rospy.wait_for_service('gazebo/set_link_state')
        self.set_link_state_srv = rospy.ServiceProxy('gazebo/set_link_state', SetLinkState)
        
        self.record_srv = rospy.Service("~record_state", RecordState, self.record_cb)
        
        self.set_srv = rospy.Service("~set_state", SetState, self.set_cb)
        
    def record_cb(self, req):
        rob_dict = {}
        # link states
        link_states = []
        for trl in self.robot_state:
            res = self.get_link_state_srv(trl[1], trl[0])
            if not res.success:
                rospy.logerr("[{}]".format(self.name))
                continue
            ls = res.link_state
            dls = state_2_dict.link_state_to_dict(ls)
            link_states.append(dls)
        path = self.save_path + '/' + req.state_name + '.yaml'
        rob_dict["link_states"] = link_states
        # model state
        model_state = self.get_model_state_srv("bd1","")
        rob_dict["model_state"] = state_2_dict.pose_to_dict(model_state.pose)
        
        with open(path, 'w') as file:
            yaml.dump(rob_dict,file)        
        return RecordStateResponse(path)
    
    def set_cb(self, req):
        with open(self.save_path + '/' + req.state_file + '.yaml', 'r') as file:
            rob_dict = yaml.load(file, Loader=yaml.FullLoader)            
            for dls in rob_dict["link_states"]:
                ls = state_2_dict.dict_to_link_state(dls)
                self.set_link_state_srv(ls)
                
            ms = ModelState()
            ms.pose = state_2_dict.dict_to_pose(rob_dict['model_state'])
            ms.pose.position.z += 0.01
            ms.model_name = 'bd1'
            self.set_model_state_srv(ms)
            return 'success'        
        
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    gsr = GazeboStateRecorder()
    gsr.run()
