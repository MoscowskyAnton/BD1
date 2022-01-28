# BD1
Gazebo ROS-based [BD1](https://starwars.fandom.com/wiki/BD-1) droid from [Star Wars: The Fallen Order](https://en.wikipedia.org/wiki/Star_Wars_Jedi:_Fallen_Order) game. For testing reinforcement learning algorithms.

![BD1 Gazebo](doc/images/bd1_gazebo.png)


# How to install
1. Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
2. Install additional ROS packages
```bash
sudo apt install ros-noetic-velocity-controllers
```
3. If you want to use GPU, install CUDA and cuDNN, [here is guide](https://medium.com/analytics-vidhya/installing-tensorflow-with-cuda-cudnn-gpu-support-on-ubuntu-20-04-f6f67745750a) _need to be followed until 'Finally to verify the correct installation'_
4. Install [pytorch](https://pytorch.org/)
5. Install [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

6. Clone this repo in your workspace and build it

# Repository structure
 - __bd1_config__ - main launches
 - __bd1_description__ - robot model
 - __bd1_gazebo_env_interface__ - utils to wrap Gazebo in Gym environment
 - __bd1_gazebo_utils__ - some Gazebo additional utilities like contact handler
 - __bd1_manual_control__ - manual robot control for testing
 - __bd1_simple_moves__ - some handmade movements like deploy\undeploy
 - __bd1_train_sb3__ - nodes implementing learning with stable-baselines3
# Launch training
```
roslaunch bd1_config bd1_gazebo.launch
```
Wait until Gazebo starts, then
```
roslaunch bd1_config bd1_train_sb3.launch 
```
Run tensorboard with 
```
tensorboard --logdir bd1_train_sb3/models/...
```
