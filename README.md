# BD1
ROS-based BD-1 droid from Star Wars The Fallen Order

![BD1 Gazebo](doc/images/bd1_gazebo.png)

## Goal of project
Make a real BD1 robot, that could walk in real world.

## Task to reach the goal
 **Achive goal in simulator first**
  - [x] Make an Gazebo controllabe model
  - [ ] Attach some RL-framework
  - [ ] Provide an evinroment interface from robot to RL-framework
  - [ ] Teach robot to deploy from conseal state without falling first
  - [ ] Teach robot to walk forward on flat surface
  - [ ] Teach robot to walk with different directions
  - [ ] Deploy some SLAM alghorithms to navigate in environments with obstacles
  - [ ] Deploy some planner
 
 **Do same for real robot**
  - [ ] Develop and construct real model 

## Getting started
Install additional ros packgaes
```
sudo apt install ros-noetic-joint-trajectory-controller
```
Install [tensorlayer](https://github.com/tensorlayer/tensorlayer)
I used tensorflow 2.2.0
