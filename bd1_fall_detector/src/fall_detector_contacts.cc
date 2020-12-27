/*
 * Example has taken from
 * https://github.com/osrf/gazebo/blob/gazebo11/examples/stand_alone/listener/listener.cc
*/

#include<string>

#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo_client.hh>

#include <iostream>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Point.h>

using std::string;

ros::Publisher fall_pub;
ros::Publisher center_of_pressure_pub;
/////////////////////////////////////////////////
// Function is called everytime a message is received.
void cb(ConstContactsPtr &_msg)
{
  std_msgs::Bool fall_msg;
  geometry_msgs::Point pressure_msg;
  if( _msg->contact_size() > 0){
    // fall detection
    for( size_t i = 0 ; i < _msg->contact_size() ; i++){
      if( _msg->contact(i).collision1() != "bd1::feet_r_link::feet_r_link_collision" &&
	  _msg->contact(i).collision1() != "bd1::feet_l_link::feet_l_link_collision"){
	//std::cout << "collision! with" << _msg->contact(i).collision1() << "\n";
	fall_msg.data = true;
	fall_pub.publish(fall_msg);
	break;
      }
    }
    // center of pressure
    float sum_x = 0;
    float sum_y = 0;
    float sum_z = 0;
    float sum_force_x = 0;
    float sum_force_y = 0;
    float sum_force_z = 0;
    //ROS_ERROR("==============");
    for( size_t i = 0 ; i < _msg->contact_size() ; i++){
      string obj1_name = _msg->contact(i).collision1();
      string obj2_name = _msg->contact(i).collision2();
      // if ground_plane is first object
      if (obj1_name.find("ground_plane") != std::string::npos) {
	for(int j = 0; j < _msg->contact(i).wrench_size(); ++j) {
	  auto pos = _msg->contact(i).position(j);
	  auto force1 = _msg->contact(i).wrench(j).body_1_wrench().force();
	  auto force2 = _msg->contact(i).wrench(j).body_2_wrench().force();
	  auto force = force1;
	  //ROS_ERROR("OBJ1 (%.4f %.4f %.4f) (%.4f %.4f %.4f)", pos.x(), pos.y(), pos.z(),
	  //	    force.x(), force.y(), force.z());
	  sum_x += pos.x() * force.x();
	  sum_y += pos.y() * force.y();
	  sum_z += pos.z() * force.z();
	  sum_force_x += force.x();
	  sum_force_y += force.y();
	  sum_force_z += force.z();
	}
      }
      // if ground_plane is second object
      if (obj2_name.find("ground_plane") != std::string::npos) {
	for(int j = 0; j < _msg->contact(i).wrench_size(); ++j) {
	  auto pos = _msg->contact(i).position(j);
	  auto force1 = _msg->contact(i).wrench(j).body_1_wrench().force();
	  auto force2 = _msg->contact(i).wrench(j).body_2_wrench().force();
	  auto force = force1;
	  //ROS_ERROR("OBJ2 (%.4f %.4f %.4f) (%.4f %.4f %.4f)", pos.x(), pos.y(), pos.z(),
	  //	    force.x(), force.y(), force.z());
	  sum_x += pos.x() * force.x();
	  sum_y += pos.y() * force.y();
	  sum_z += pos.z() * force.z();
	  sum_force_x += force.x();
	  sum_force_y += force.y();
	  sum_force_z += force.z();
	}
      }
      
    }
    pressure_msg.x = sum_x / sum_force_x;
    pressure_msg.y = sum_y / sum_force_y;
    pressure_msg.z = sum_z / sum_force_z;
  }
  fall_pub.publish(fall_msg);
  center_of_pressure_pub.publish(pressure_msg);
}

/////////////////////////////////////////////////
int main(int _argc, char **_argv)
{  

  ros::init(_argc, _argv, "robot_fall_detector");      
  ros::NodeHandle nh_p("~");
  
  fall_pub = nh_p.advertise<std_msgs::Bool>("fall",1);
  center_of_pressure_pub = nh_p.advertise<geometry_msgs::Point>("center_of_pressure",1);
 
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
