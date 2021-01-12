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
#include <bd1_gazebo_utils/FeetContacts.h>

ros::Publisher fall_pub;
ros::Publisher feet_contacts_pub;

// determinant of 3x3 matrix
/*
float det(float c00, float c01, float c02,
	  float c10, float c11, float c12,
	  float c20, float c21, float c22){
  return c00 * (c11*c22 - c21*c12) - c01 * (c10*c22 - c12*c20) + c02 * (c10*c21 - c11*c20);
}
*/

/////////////////////////////////////////////////
// Function is called everytime a message is received.
void cb(ConstContactsPtr &_msg)
{
  std_msgs::Bool fall_msg;
  bd1_gazebo_utils::FeetContacts feet_contacts_msg;

  // center of pressure is the point r=(x,y,z) where sum of torques is zero:
  // sum [Fi, ri - r] = 0;
  
  //      |     i       j      k    |
  // sum (| rix - x riy - y riz - z |) = 0;
  //      |    Fix     Fiy    Fiz   |

  // if rix, riy, riz are not restricted then:
  
  // sum (Fiy (riz - z) - Fiz (riy - y)) = 0
  // sum (Fix (riz - z) - Fiz (rix - x)) = 0
  // sum (Fix (riy - y) - Fiy (rix - x)) = 0

  // and:
  
  // y sum(Fiz) - z sum(Fiy) = sum(Fiz riy - Fiy riz)
  // x sum(Fiz) - z sum(Fix) = sum(Fiz rix - Fix riz)
  // x sum(Fiy) - y sum(Fix) = sum(Fiy rix - Fix riy)

  // but riz === 0 => z === 0 therefore this system should be simplified:

  // sum (Fiz (riy - y) = 0
  // sum (Fiz (rix - x) = 0

  // x = sum(Fiz rix) / sum(Fiz)
  // y = sum(Fiz riy) / sum(Fiz)
  
  std::string str_state = "\n";
  char str_data[10000];
  if( _msg->contact_size() > 0){
    float sum_f_z = 0;
    float sum_1 = 0, sum_2 = 0;
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
	    sprintf(str_data, "%s:\n", _msg->contact(i).collision1().c_str());
	    str_state += str_data;
	    for (int j=0; j< _msg->contact(i).wrench().size(); ++j) {
	      float fx = _msg->contact(i).wrench(j).body_1_wrench().force().x();
	      float fy = _msg->contact(i).wrench(j).body_1_wrench().force().y();
	      float fz = _msg->contact(i).wrench(j).body_1_wrench().force().z();
	      float rx = _msg->contact(i).position(j).x();
	      float ry = _msg->contact(i).position(j).y();
	      float rz = _msg->contact(i).position(j).z();
	      sum_f_z += fz;
	      sum_1 += fz*ry;
	      sum_2 += fz*rx;
	      //sprintf(str_data, "  (%.3f,%.3f,%.3f) - (%.3f,%.3f,%.3f)\n", fx, fy, fz, rx, ry, rz);
	      //str_state += str_data;

	    }
	  }
      }
      
      // calculate center of pressure
      /* if x,y,z are not restricted:
      sprintf(str_data, "sum = (%.3f,%.3f,%.3f)\n", sum_f_x, sum_f_y, sum_f_z);
      str_state += str_data;
      sprintf(str_data, "sum2 = (%.3f,%.3f,%.3f)\n", sum_1, sum_2, sum_3);
      str_state += str_data;
      float d = det(      0,  sum_f_z, -sum_f_y,
		    sum_f_z,        0, -sum_f_x,
                    sum_f_y, -sum_f_x,        0);
      float dx = det(sum_1,  sum_f_z, -sum_f_y,
		     sum_2,        0, -sum_f_x,
          	     sum_3, -sum_f_x,        0);
      float dy = det(      0,  sum_1, -sum_f_y,
	             sum_f_z,  sum_2, -sum_f_x,
                     sum_f_y,  sum_3,        0);
      float dz = det(      0,  sum_f_z, sum_1,
		     sum_f_z,        0, sum_2,
                     sum_f_y, -sum_f_x, sum_3);
      sprintf(str_data, "d = (%.3f,%.3f,%.3f,%.3f)\n", d, dx, dy, dz);
      str_state += str_data;
      */
      sprintf(str_data, "res = (%.3f,%.3f,%.3f)\n", sum_1/sum_f_z, sum_2/sum_f_z, 0);
      str_state += str_data;
  }
  ROS_ERROR(str_state.c_str());
  
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
