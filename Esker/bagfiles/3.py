import rosbag
bag = rosbag.Bag('2020.bag')
for topic, msg, t in bag.read_messages(topics=['/tf', '/rosout']):
    print(msg)
bag.close();
