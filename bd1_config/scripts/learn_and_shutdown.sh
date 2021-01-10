#!/bin/bash
roslaunch bd1_config bd1_gazebo.launch
sudo -S shutdown now <<< $1
