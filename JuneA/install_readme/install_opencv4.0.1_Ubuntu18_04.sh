sudo apt-get install build-essential cmake pkg-config yasm -y
sudo apt-get install libjpeg-dev libtiff5-dev libpng-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev -y
sudo apt-get install libv4l-dev v4l-utils -y
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev -y
sudo apt-get install libqt4-dev -y
sudo apt-get install libatlas-base-dev gfortran libeigen3-dev -y
sudo apt-get install python3-dev python3-numpy -y
sudo apt-get install python-dev python-numpy -y
sudo apt-get install libblas-dev liblapack-dev -y
sudo apt-get install libpng16-16 libtiff5-dev libgst-dev libgstreamer1.0-dev libglib2.0-dev -y
sudo apt-get install opencl-headers  ocl-icd-libopencl1 -y
sudo apt-get install mesa-utils libgl1-mesa-dri libqt4-opengl-dev -y
sudo apt-get install libatlas-base-dev gfortran libeigen3-dev -y
sudo apt-get install libhdf5-dev -y
sudo apt-get install libgtk-3-dev -y
sudo apt-get install libgtk2.0-dev -y
sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev -y

cd ~
if [ -d lib]
then
	echo "lib directory check"
else
	echo "lib directory don't check"
	mkdir lib
fi
mkdir ~/lib/opencv
cd ~/lib/opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.0.1.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.0.1.zip
unzip opencv_contrib.zip
cd ~/lib/opencv/opencv-4.0.1/
mkdir build
cd ~/lib/opencv/opencv-4.0.1/build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=OFF \
-D WITH_IPP=OFF \
-D WITH_1394=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_DOCS=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D WITH_QT=OFF \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.0.1/modules \
-D WITH_V4L=ON  \
-D WITH_FFMPEG=ON \
-D WITH_XINE=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON ../

##-- SUCCESS
##-- Configuring done
##-- Generating done
##-- Build files have been written to: /home/webnautes/opencv/opencv-4.0.1/build

time make -j4
sudo make install
echo "########################find /usr/local/lib###############################"
cat /etc/ld.so.conf.d/* |grep /usr/local/lib
