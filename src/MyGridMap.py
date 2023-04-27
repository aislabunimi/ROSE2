#ros library
import roslib
import rospy
import copy
from nav_msgs.msg import OccupancyGrid

#Other Libary

import copy
import io
from PIL import Image

class MyGridMap:
	def __init__(self,grid):
		self.occupancyGrid = grid
		self.mapWidth = grid.info.width
		self.mapHeight = grid.info.height
		self.lethalCost = 100 #Value for the cell with an obstacle. 0 Free cell, -1 unknown
	
	def getWidth(self):
		return self.mapWidth
	
	def getHeight(self):
		return self.mapHeight
	
	def getSize(self):
		return self.mapHeight * self.mapWidth
	
	def getResolution(self):
		return self.occupancyGrid.info.resolution
	
	def getOriginX(self):
		return self.occupancyGrid.info.origin.position.x
	
	def getOriginY(self):
		return self.occupancyGrid.info.origin.position.y
	
	def getOccupancyGrid(self):
		return self.occupancyGrid
	
	def getCoordinates(self,index):
		if (index > (self.mapWidth * self.mapHeight)):
			print ("Error, index too big!")
			return -1, -1
		else:
			x = index/ self.mapWidth
			y = index % self.mapWidth
			return x,y
	
	def getIndex(self,x,y):
		if(x > self.mapWidth or y > self.mapHeight):
			print ("Error, get index failed")
			return -1
		else:
			return (y * self.mapWidth + x)
	
	def getData(self,index):
		if(index < (self.mapWidth * self.mapHeight) and index > 0):
			return self.occupancyGrid.data[index]
		else:
			print ("Error, wrong index")
			return -1
	
	def getData(self,x,y):
		if(x < 0 or x > self.mapWidth or y < 0 or y > self.mapHeight):
			return -1
		return self.occupancyGrid.data[y * self.mapWidth + x]
	
	def isFree(self,index):
		value = self.getData(index)
		if(value >= 0 and value < self.lethalCost):
			return True
		else:
			return False
		

	def isFree(self,x,y):
		if(x < 0 or x > self.mapWidth or y < 0 or y > self.mapHeight):
			return False
		else:
			value = self.getData(x,y)
			if(value >= 0 and value < self.lethalCost):
				return True
			else:
				return False
			
	def isFrontier(self,index):
		x = index / self.mapWidth
		y = index % self.mapWidth
		if(self.getData(x-1,y-1) == -1):
			return True
		if(self.getData(x-1,y  ) == -1):
			return True
		if(self.getData(x-1,y+1) == -1):
			return True
		if(self.getData(x  ,y-1) == -1):
			return True
		if(self.getData(x  ,y+1) == -1):
			return True
		if(self.getData(x+1,y-1) == -1):
			return True
		if(self.getData(x+1,y  ) == -1):
			return True
		if(self.getData(x+1,y+1) == -1):
			return True
		return False
	
	def isFrontier(self,x,y):
		i = self.getIndex(x,y)
		if( i == -1):
			print("Wrong coordinate")
			return False
		else:
			return self.isFrontier(i)
		
	def createPGM(self):
		my_file = io.BytesIO()
		pgmHeader ='P5' + '\n' + ' ' + str(self.mapWidth) + ' ' + str(self.mapHeight) + '\n' + ' ' + str(255) +  '\n'
		pgmHeader_byte = bytearray(pgmHeader,'utf-8')
		my_file.write(pgmHeader_byte)
		for y in range (0,self.mapHeight):
			for x in range (0,self.mapWidth):
				index = x + (self.mapHeight-y-1)*self.mapWidth
				value = self.occupancyGrid.data[index]
				if value == +100:
					my_file.write((0).to_bytes(1,'big'))
				elif value >= 0 and value <= self.lethalCost:
					my_file.write((254).to_bytes(1,'big'))
				else:
					my_file.write((205).to_bytes(1,'big'))
		my_file.truncate()
		my_file.seek(0)
		t_image = Image.open(my_file)
		image = t_image.copy()
		my_file.close()
		return image
	
		
	def areNeighbour(self,index1,index2):		
		x1,y1 = self.getCoordinates(index1)
		x2,y2 = self.getCoordinates(index2)
		if( x1 == -1 or x2 == -1):
			print("Error, wrong Index!")
			return False
		if(x1 == x2 and y1 == y2): 
			return True
		if(x1 == (x2-1) and y1 == (y2-1)):
			return True
		if(x1 == x2 and y1 == (y2-1)):
			return True
		if(x1 == (x2+1) and y1 == (y2-1)):
			return True
		if(x1 == (x2-1) and y1 == y2):
			return True
		if(x1 == (x2+1) and y1 == y2):
			return True
		if(x1 == (x2-1) and y1 == (y2+1)):
			return True
		if(x1 == x2 and y1 == (y2+1)):
			return True
		if(x1 == (x2+1) and y1 ==(y2+1)):
			return True
		return False

