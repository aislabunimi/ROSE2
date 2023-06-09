#! /usr/bin/env python
import rospy
import numpy as np
from rose2.msg import ROSEFeatures
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from skimage.util import img_as_ubyte
from util import MsgUtils as mu
from PIL import Image as ImagePIL
from rose_v1_repo.fft_structure_extraction import FFTStructureExtraction as structure_extraction
import rose2.srv
import warnings
"""
Subscribers: /map (nav_msgs/OccupancyGrid)
Publishers: /features_ROSE (rose/ROSEFeatures)
            /clean_map (nav_msgs/OccupancyGrid)
            /direction_markers (visualization_msgs/MarkerArray)
Service:    /features_roseSrv (rose2/ROSE)
params:     /filter -> used for rose method (real number 0 to 1, 0 keeps all the pixels)
            /pub_once -> publish once or keep publishing at a certain rate (True or False)
            
This node gets a map and applies ROSE method to it. This method is used to find the main directions of the map,
and to filter all the pixels which do not belong to the main structure.  
After computing the result rose publishes ROSEFeatures: main_directions, originalMap and cleanMap
The cleanMap corresponds to the original map with non-structural pixels removed, that is, only what is considered to be
a wall or part of it is kept in the clean map.
"""
class FeatureExtractorROSE:
    def __init__(self):
        rospy.init_node('ROSE')
        rospy.loginfo('[ROSE] waiting for map...')
        #ROS params
        self.pubOnce = rospy.get_param("ROSE/pub_once", True)
        self.filter = rospy.get_param("ROSE/filter", 0.18)

        #features storing variables
        self.features = None
        self.origin = None
        self.res = None
        self.imgMap = None
        self.rose = None
        self.cleanMapOcc = None
        #flow control var
        self.publish = False # True if this node has new features to publish

        self.pubRate = 1

        #Service
        rospy.Service('ROSESrv', rose2.srv.ROSE, self.featuresSrv)

        #PUBLISHERS
        self.pubFeatures = rospy.Publisher('features_ROSE', ROSEFeatures, queue_size=1)
        self.pubCleanMap = rospy.Publisher('clean_map', OccupancyGrid, queue_size=1)
        self.pubDirections = rospy.Publisher('direction_markers', MarkerArray, queue_size=1)
        #SUBSCRIBER
        rospy.Subscriber('map', OccupancyGrid, self.processMap, queue_size=1, buff_size=2**28)

    def run(self):
        r = rospy.Rate(self.pubRate)
        while not rospy.is_shutdown():
            if self.features is not None and self.publish:
                self.publishFeatures()
                if self.pubOnce :
                    self.publish = False
            r.sleep()

    #CALLBACK
    def processMap(self, occMap):
        self.origin = occMap.info.origin
        self.res = occMap.info.resolution
        self.imgMap = mu.fromOccupancyGridToImg(occMap)
        grid_map = img_as_ubyte(ImagePIL.fromarray(self.imgMap))
        self.rose = structure_extraction(grid_map, peak_height=0.2, par=50)
        self.rose.process_map()
        if len(self.rose.main_directions) > 2:
            self.rose.simple_filter_map(self.filter)
            self.rose.generate_initial_hypothesis_simple()
            try:
                self.rose.find_walls_flood_filing()
            except ValueError:
                rospy.loginfo("[ROSE] skipped data, can't find walls...")
                return
            self.analysed_map_uint8 = self.rose.analysed_map.astype(np.uint8)
            self.main_directions = self.rose.main_directions
            self.cleanMapOcc = mu.fromImgMapToOccupancyGridRaw(self.analysed_map_uint8, self.origin, self.res)
            self.features = ROSEFeatures(occMap, self.cleanMapOcc, self.main_directions)
            rospy.loginfo("[ROSE] DONE.")
            self.publish = True
            rospy.loginfo('[ROSE] publishing clean map and main directions')
        else:
            rospy.loginfo('[ROSE] skipped data, not enough directions computed...')
            return
    def featuresSrv(self, occMap):
        if(self.features is None):
            self.processMap(occMap)
        return self.features


    def publishFeatures(self):
        self.pubFeatures.publish(self.features)
        self.pubCleanMap.publish(self.cleanMapOcc)
        self.pubDirections.publish(mu.make_direction_markers(self.main_directions, self.origin, self.res))
if __name__ == '__main__':
    f = FeatureExtractorROSE()
    warnings.filterwarnings("ignore")
    f.run()
