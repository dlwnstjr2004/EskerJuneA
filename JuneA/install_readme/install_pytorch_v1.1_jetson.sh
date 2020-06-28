## pytorch v1.1
sudo apt-get install python3-pip python3-dev -y
python3 -m pip install --upgrade pip
sudo apt-get install libjpeg-dev zlib1g-dev -y

wget https://nvidia.box.com/shared/static/mmu3xb3sp4o8qg9tji90kkxl1eijjfc6.whl –o torch-1.1.0-cp36-cp36m-linux_aarch64.whl

pip3 install numpy torch-1.1.0-cp36-cp36m-linux_aarch64.whl
## torchvision v0.3.0
sudo apt-get install libjpeg-dev zlib1g-dev -y
git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
cd torchvision
(sudo) python setup.py install
