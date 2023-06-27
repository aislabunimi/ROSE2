import array
import pickle

import rospy
from rose2.msg import ROSE2Features
from pprint import pprint


#Node to debug FeatureExtractorrose2. It receive features from that node, unpickles them and prints them.
class ReceiverFeatures:
    def __init__(self):
        rospy.init_node('FeatureReceiver')
        print('Waiting for features...')
        rospy.Subscriber('features_ROSE2', ROSE2Features, self.processFeatures)
        rospy.spin()

    def processFeatures(self, features):
        print('EXTENDED LINES')
        for l in features.lines:
            line = pickle.loads(array.array('b',l.bytes))
            pprint(vars(line))
        print('EDGES')
        for e in features.edges:
            edge = pickle.loads(array.array('b',e.bytes))
            pprint(vars(edge))
        print('ROOMS')
        for r in features.rooms:
            print(pickle.loads(array.array('b',r.bytes)))
        print('CONTOUR')
        print(features.contour)

if __name__ == '__main__':

    f = ReceiverFeatures()
