# Vulkan

- 다양한 플렛폼에서 사용가능
- Graphics look nicer, and games run faster, on both DirectX 12 and Vulkan

wget https://sdk.lunarg.com/sdk/download/1.1.85.0/linux/vulkansdk-linux-x86_64-1.1.85.0.tar.gz

tar xzvf vulkansdk-linux-x86_64-1.1.85.0.tar.gz

sudo apt-get update

sudo apt-get dist-upgrade -y

sudo apt-get install libglm-dev cmake libxcb-dri3-0 libxcb-present0 libpciaccess0 \
libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev \
libmirclient-dev libwayland-dev libxrandr-dev libxcb-ewmh-dev -y
sudo apt install libxcb1-dev xorg-dev -y

sudo add-apt-repository ppa:oibaf/graphics-drivers
sudo apt-get install vulkan-utils mesa-vulkan-drivers -y

(Ubuntu18.04)
sudo apt-get install qt5-default qtwebengine5-dev -y

sudo apt-get install git libpython2.7 -y

(~/yourpath/1.2.131.2/)
source setup-env.sh

(~/yourpath/1.2.131.2/examples)
mkdir build
cd build
cmake ..
make -j4



