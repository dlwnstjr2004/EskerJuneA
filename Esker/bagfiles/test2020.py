import rosbag

#searched data will push "search.bag" file
from std_msgs.msg import Int32, String

f = open('search.txt', mode='w')

bag = rosbag.Bag('2020.bag')
for topic, msg, t in bag.read_messages(topics=['/slam_cloud']):
   f.write(msg)
bag.close()
f.close()
