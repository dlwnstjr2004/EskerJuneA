##Dependecy for tensorflow on jetPack 4.3

sudo apt-get update -y
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev -y
sudo apt-get install wget -y

#Install and upgrade pip3
sudo apt-get install python3-pip python3-dev
sudo pip3 install -U pip testresources setuptools

#Install the Python package dependencies
sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 enum34 futures protobuf

# enter the link
# https://developer.download.nvidia.com/compute/redist/jp/v43
# download tensorflow_gpu-1.15.0+nv19.......whl
cd ~/Downloads
wget https://developer.download.nvidia.com/compute/redist/jp/v43/tensorflow-gpu/tensorflow_gpu-1.15.0+nv20.1-cp36-cp36m-linux_aarch64.whl
pip3 install --user tensorflow_gpu-1.15.0+nv20.1-cp36-cp36m-linux_aarch64.whl

#check your tensorflow version
python3 -c 'import tensorflow as tf;print(tf.__version__)'

