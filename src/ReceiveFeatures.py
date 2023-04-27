import array
import pickle

import rospy
from feature_extraction.msg import rose2Features


#Node to debug FeatureExtractorrose2. It receive features from that node, unpickles them and prints them.
class ReceiverFeatures:
    def __init__(self):
        rospy.init_node('FeatureReceiver')
        print('Waiting for features...')
        rospy.Subscriber('features_rose2', rose2Features, self.processFeatures)
        rospy.spin()

    def processFeatures(self, features):
        print('EXTENDED LINES')
        for l in features.lines:
            print(pickle.loads(array.array('b',l.bytes)))
        print('EDGES')
        for e in features.edges:
            print(pickle.loads(array.array('b',e.bytes)))
        print('ROOMS')
        for r in features.rooms:
            print(pickle.loads(array.array('b',r.bytes)))
        print('CONTOUR')
        print(features.contour)

if __name__ == '__main__':

    f = ReceiverFeatures()