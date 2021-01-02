/*
 * Example has taken from
 * https://github.com/osrf/gazebo/blob/gazebo11/examples/stand_alone/listener/listener.cc
*/

#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>

#include <iostream>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <bd1_gazebo_utils/FeetContacts.h>

ros::Publisher fall_pub;
ros::Publisher feet_contacts_pub;
/////////////////////////////////////////////////
// Function is called everytime a message is received.
void cb(ConstContactsPtr &_msg)
{
  std_msgs::Bool fall_msg;
  bd1_gazebo_utils::FeetContacts feet_contacts_msg;
  if( _msg->contact_size() > 0){      
      for( size_t i = 0 ; i < _msg->contact_size() ; i++){
          if( _msg->contact(i).collision2() == "ground_plane::link::collision"){
              
            if( _msg->contact(i).collision1() == "bd1::foot_r_link::foot_r_link_collision" )
                feet_contacts_msg.foot_r = true;
            else if( _msg->contact(i).collision1() == "bd1::foot_l_link::foot_l_link_collision" )
                feet_contacts_msg.foot_l = true;
            else if( _msg->contact(i).collision1() == "bd1::foot_l_link::foot_l_link_fixed_joint_lump__heel_l_link_collision_1" )
                feet_contacts_msg.heel_l = true;
            else if( _msg->contact(i).collision1() == "bd1::foot_r_link::foot_r_link_fixed_joint_lump__heel_r_link_collision_1" )        
                feet_contacts_msg.heel_r = true;
            else{            
                fall_msg.data = true;                                
            }
        }
      }
  }
  fall_pub.publish(fall_msg);
  feet_contacts_pub.publish(feet_contacts_msg);
}

/////////////////////////////////////////////////
int main(int _argc, char **_argv)
{  

  ros::init(_argc, _argv, "robot_fall_detector");      
  ros::NodeHandle nh_p("~");
  
  fall_pub = nh_p.advertise<std_msgs::Bool>("fall",1);
  feet_contacts_pub = nh_p.advertise<bd1_gazebo_utils::FeetContacts>("feet_contacts",1);
 
  // Load gazebo
  gazebo::client::setup(_argc, _argv);

  // Create our node for communication
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();

  // Listen to Gazebo world_stats topic
  gazebo::transport::SubscriberPtr sub = node->Subscribe("/gazebo/default/physics/contacts", cb);
  
  // Busy wait loop...replace with your own code as needed.
  ros::spin();
//   while (true)
//     gazebo::common::Time::MSleep(10);

  // Make sure to shut everything down.
  gazebo::client::shutdown();
}
