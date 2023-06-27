# ROSE2
<p>This package provides ROS nodes to execute ROSE and ROSE2 methods.</p>

To make it run build the repository in your ROS workspace, install packages from requirements.txt then type in terminal: 
```
roslaunch rose2 ROSE.launch 
```
This will start two nodes: ROSE and ROSE2. To use them publish an OccupancyGrid map in the /map topic

Additional requirements to install via apt:
```
ros-VERSION-jsk-recognition-msgs
ros-VERSION-jsk-rviz-plugins
```
