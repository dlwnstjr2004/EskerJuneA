git clone https://github.com/JetsonHacksNano/installROS
cd installROS/
./installROS.sh -p ros-melodic-desktop
#Installation complete!
./setupCatkinWorkspace.sh
gedit ~/.bashrc
# check ros ip
source ~/.bashrc
