import array

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32
from visualization_msgs.msg import Marker
from rose2.msg import Edge
from rose2.msg import Room
from rose2.msg import Contour
from rose2.msg import ExtendedLine
from scipy.spatial.transform import Rotation
import numpy as np
import random
import rospy
import pickle
from MyGridMap import MyGridMap
import rospy
from matplotlib import pyplot as plt
from PIL import Image
import cv2
#Utility file used to :
# - convert messages to data structures and viceversa
# - create messages

"""
OccupancyGrid -> np.array
Use this if the OccupancyGrid is in the standard form: values are 0,-1 or 100 (mode: trinary of yaml format)
"""
def fromOccupancyGridToImg(occupancyGrid):
    gridMap = MyGridMap(occupancyGrid)
    map = np.zeros((gridMap.mapHeight, int(gridMap.mapWidth)))
    height, width = map.shape
    for i in range(height):
        for j in range(width):
            val = gridMap.getData(j, i)
            if val == 0:
                map[i,j] = 255
            if val == -1:
                map[i,j] = 200
            if val == 100:
                map[i,j] = 0
    map = np.array(map)
    map = (map).astype(np.uint8)
    return map
"""
OccupancyGrid  -> np.array
Use this if the OccupancyGrid has pixel values between 0 and 255 (that corresponds to mode: raw of yaml format)
"""
def fromOccupancyGridRawToImg(occupancyGrid):
    gridMap = MyGridMap(occupancyGrid)
    map = np.zeros((gridMap.mapHeight, int(gridMap.mapWidth)))
    height, width = map.shape
    for i in range(height):
        for j in range(width):
            map[i, j] = gridMap.getData(j, i)
    map = np.array(map)
    map = (map).astype(np.uint8)
    return map

# np.array -> OccupancyGrid wth raw values [0,255]
def fromImgMapToOccupancyGridRaw(imgMap, origin,res):
    grid = OccupancyGrid()
    grid.header = Header()
    grid.header.frame_id = "map"
    grid.info = MapMetaData()
    grid.info.resolution = res
    grid.info.height = imgMap.shape[0]
    grid.info.width = imgMap.shape[1]
    grid.data = list(imgMap.copy().ravel())
    grid.info.origin = origin
    return grid

# ExtendedSegment -> ExtendedLine.msg
def make_extendedline_msg(line):
    extLineMsg = ExtendedLine()
    d = array.array('b', (pickle.dumps(line)))
    extLineMsg.bytes = d
    return extLineMsg

# Segment -> Edge.msg
def make_edge_msg(e):
    edgeMsg = Edge()
    d = array.array('b', (pickle.dumps(e)))
    edgeMsg.bytes = d
    return edgeMsg
# room (shapely polygon) -> Room.msg
def make_room_msg(room):
    roomMsg = Room()
    d = array.array('b',(pickle.dumps(room)))
    roomMsg.bytes = d
    return roomMsg
# array of float32 -> Contour
def make_contour_msg(vertices):
    contour = Contour()
    contour.vertices = np.ravel(vertices)
    return contour

# Segment, id -> Marker
def make_edge_marker(edge, i, origin,res):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "EDGES-%u" % 0
    marker.id = i
    marker.type = 5
    marker.action = 0
    marker.pose.position = origin.position
    marker.scale.x = res
    marker.color.a = 1
    marker.color.r = random.uniform(0,1)
    marker.color.g = random.uniform(0,1)
    marker.color.b = random.uniform(0,1)
    p1 = Point()
    p2 = Point()
    p1.x = edge.x1*res
    p1.y = edge.y1*res
    p2.x = edge.x2*res
    p2.y = edge.y2*res
    marker.points.append(p1)
    marker.points.append(p2)
    return marker

def make_lines_marker(lines, origin,res):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "LINES"
    marker.id = 50
    marker.type = 5
    marker.action = 0
    marker.pose.position = origin.position
    marker.scale.x = res
    marker.color.a = 1
    marker.color.r = 1
    for l in lines:
        p1 = Point()
        p2 = Point()
        p1.x = l.x1*res
        p1.y = l.y1*res
        p2.x = l.x2*res
        p2.y = l.y2*res
        marker.points.append(p1)
        marker.points.append(p2)
    return marker
def make_room_polygon(room, origin, res):
    origin_x = origin.position.x
    origin_y = origin.position.y
    points = []
    points_x, points_y = room.exterior.coords.xy
    for i in range(len(points_x)):
        p = Point32()
        p.x = points_x[i]*res + origin_x
        p.y = points_y[i]*res + origin_y
        points.append(p)
    polygon = Polygon()
    polygon.points = points
    polygonStamped = PolygonStamped()
    polygonStamped.polygon = polygon
    polygonStamped.header.frame_id = "map"
    polygonStamped.header.stamp = rospy.Time.now()
    return polygonStamped
def make_direction_markers(main_directions, origin, res):
    directions = []
    markers = []
    for i in range(0, len(main_directions), 2):
        directions.append((main_directions[i], main_directions[i + 1]))
        i = 0
    i = 0
    for d in directions:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "DIR-%u" % i
        marker.id = i
        marker.type = 0
        marker.action = 0
        #marker.pose.position = origin.position

        rot = Rotation.from_euler('xyz', [d[0], 0, d[1]])
        rot_quat = rot.as_quat()
        marker.pose.orientation = Quaternion(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
        marker.scale.x = 5
        marker.scale.y = res
        marker.scale.z = res
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        i += 1
        markers.append(marker)
    return markers
