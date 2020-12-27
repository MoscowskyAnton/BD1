# BD1
ROS-based BD-1 droid from [Star Wars: The Fallen Order](https://en.wikipedia.org/wiki/Star_Wars_Jedi:_Fallen_Order) game.

![BD1 Gazebo](doc/images/bd1_gazebo.png)

# Goal of project
Make a real BD1 robot, that could walk in the real world envinroment.

# Tasks to reach the goal
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

# Tasks roadmap
<body>
  Here is a mermaid diagram:
  <div class="mermaid">
    graph TD
    A[Client] --> B[Load Balancer]
    B --> C[Server01]
    B --> D[Server02]
  </div>
</body>

# How to install
0. Of cource you need installed ROS Noetic (need for python3)
1. Install additional ROS packages (list is not full, because hasn't been tested on virgin mashine yet)
```bash
sudo apt install ros-noetic-joint-trajectory-controller
sudo apt install ros-noetic-velocity-controllers
```
2. Install [tensorlayer](https://github.com/tensorlayer/tensorlayer)
```bash
pip3 install tensorflow-gpu==2.0.0-rc1
```
**or** if you has no GPU
```bash
pip3 install tensorflow==2.2.0
```
then
```bash
pip3 install tensorlayer
pip3 install tensorflow_probability==0.10.1 # for TD3 usage
```
Version 2.2.0 is newest version that throws no errors for me.

# Repository structure
 - 
