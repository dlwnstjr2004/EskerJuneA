sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://pgp.mit.edu:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA117
sudo apt-get install ros-melodic-desktop
sudo rosdep init
sudo rosdep update
source /opt/ros/melodic/setup.bash
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential -y
sudo apt-get install python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
roscore
sudo -H pip3 install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg
