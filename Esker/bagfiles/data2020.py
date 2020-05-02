import rosbag

bag = rosbag.Bag('2020.bag')
for topic, msg, t in bag.read_messages(topics=['/slam_cloud']):
    print(msg)#f.write(msg)
bag.close()
