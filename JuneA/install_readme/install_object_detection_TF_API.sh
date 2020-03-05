##Dependecies for object_detection

sudo apt-get install python3-dev python3-pip -y
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk -y
sudo apt-get install libpng-dev libfreetype6-dev -y

pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user pillow
pip3 install --user lxml
pip3 install --user jupyter
pip3 install --user matplotlib

cd ~
if [ -d work]
then
	echo "work directory check"
else
	echo "work directory don't check"
	mkdir work
fi

cd ~/work
if [ -d git]
then
	echo "git directory check"
else
	echo "git directory don't check"
	mkdir git
fi

cd git
# From ~/work/git/
git clone http://github.com/tensorflow/models

# install cocoapi
# git clone http://github.com/cocodataset/cocoapi.git
# cp cocoapi/PythonAPI/pycocotools /models/research

cd models/research

protoc object_detection/protos/*.proto --python_out=.

# From ~/<your PATH>/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# try
python3 object_detection/builders/model_builder_test.py

# objectdetection_tutorial on jupyter notebook 
# 
