# BD1
ROS-based BD-1 droid from [Star Wars: The Fallen Order](https://en.wikipedia.org/wiki/Star_Wars_Jedi:_Fallen_Order) game.

![BD1 Gazebo](doc/images/bd1_gazebo.png)

## Goal of project
Make a real BD1 robot, that could walk in the real world envinroment.

## Task to reach the goal
 **Achive goal in simulator first**
  - [x] Make an Gazebo controllabe model
  - [x] Attach some RL-framework
  - [x] Provide an evinroment interface from robot to RL-framework
  - [ ] Teach robot to deploy from conseal state without falling first
  - [ ] Teach robot to walk forward on flat surface
  - [ ] Teach robot to walk with different directions
  - [ ] Deploy some SLAM alghorithms to navigate in environments with obstacles
  - [ ] Deploy some planner
 
 **Do same for real robot**
  - [ ] Develop and construct real model 

## Install this
0. Ofcource you need installed ROS Noetic (need for python3)
1. Install additional ROS packages
```bash
sudo apt install ros-noetic-joint-trajectory-controller
```
2. Install [tensorlayer](https://github.com/tensorlayer/tensorlayer)
```bash
pip3 install tensorflow==2.2.0
pip3 install tensorlayer
```
Version 2.2.0 is newest version that throws no errors for me.
