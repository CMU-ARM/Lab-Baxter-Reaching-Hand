#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import roslib
from roslib import message
roslib.load_manifest('test_cam')
import sys
import rospy
import cv2
import math
import copy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import baxter_interface
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import time
import threading
import struct
from baxter_interface import settings
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
import argparse
from sensor_msgs.msg import (
    JointState
)
from baxter_core_msgs.msg import (
    JointCommand,
    EndpointState,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    PointStamped,
    Quaternion,
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from std_msgs.msg import (
    Float64,
    Header,
)
# TODO (amal): clean up these imports!!!!

class Coord(object):
    def __init__(self, x, y, z, now):
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            raise Exception("Trying to create coord with nans...%f, %f, %f" % (x,y,z))
        self.x = x
        self.y = y
        self.z = z
        self.now = now

    def __str__(self):
        return "Coord(x="+str(self.x)+", y="+str(self.y)+", z="+str(self.z)+", now="+str(self.now)+")"

    def __repr__(self):
        return self. __str__()

class Rectangle(object):
    def __init__(self, x, y, w, h, now):
        if math.isnan(x) or math.isnan(y) or math.isnan(w) or math.isnan(h):
            raise Exception("Trying to create coord with nans...%f,%f,%f,%f" % (x,y,w,h))
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.now = now

    def __str__(self):
        return "Rect(x="+str(self.x)+", y="+str(self.y)+", w="+str(self.w)+", h="+str(self.h)+", now="+str(self.now)+")"

    def __repr__(self):
        return self. __str__()

    def getCenter(self):
        return (self.x+self.w/2, self.y+self.h/2)

class Hand(object):
    def __init__(self, point, maxDx, maxDy, maxDz):
        if type(point) != PointStamped:
            raise TypeException("first argument to hand must be of type rectangle")
        self.positions = [point]
        self.maxDx = maxDx
        self.maxDy = maxDy
        self.maxDz = maxDz
        # self.avgX, self.avgY, self.avgW, self.avgH = None, None, None, None

    def __str__(self):
        string = "[\n"
        for point in self.positions:
            string += "     "+str(point)+"\n"
        return string+"]"

    def __repr__(self):
        return self. __str__()

    def couldHaveMovedHere(self, point):
        return (abs(point.point.x-self.positions[-1].point.x) < self.maxDx and
        abs(point.point.y-self.positions[-1].point.y) < self.maxDy and
        abs(point.point.z-self.positions[-1].point.z) < self.maxDz)

    # returns whether the rect was appended (if the hand could viably move there).
    # if the average rect was previously calculated, it also returns
    def addPosition(self, point):
        # if rect.now is None:
        #     # return None, None, False
        #     return False
        if self.couldHaveMovedHere(point):
            self.positions.append(point)
            # if self.avgX is None: # If we haven't gotten an average yet
            #     return None, None, True
            # else: # if we have gotten an average
            #     avgMidX = self.avgX+self.avgW/2
            #     avgMidY = self.avgY+alf.avgH/2
            #     midX = rect.x+rect.w/2
            #     midY = rect.y+rect.h/2
            #     return midX-avgMidX, midY-avgMidY, True
            return True
        # return None, None, False
        return False

    def getLatestPos(self):
        if len(self.positions) == 0:
            return None
        return self.positions[-1]

    # Gets the average of the last nTimes positions the hand has been in
    # def getAveragePosByNum(self, nTimes):
    #     if nTimes <= 0 or len(self.positions) == 0:
    #         return None
    #     num = min(nTimes, len(self.positions))
    #     avgX, avgY, avgW, avgH = 0, 0, 0, 0
    #     for i in xrange(len(self.positions)-num, len(self.positions)):
    #         avgX += self.positions[i].x
    #         avgY += self.positions[i].y
    #         avgW += self.positions[i].w
    #         avgH += self.positions[i].h
    #     avgX /= num
    #     avgY /= num
    #     avgW /= num
    #     avgH /= num
    #     # self.avgX = avgX
    #     # self.avgY = avgY
    #     # self.avgW = avgW
    #     # self.avgH = avgH
    #     return Rectangle(avgX, avgY, avgW, avgH, None), nTimes

    # Gets the average of the last positions that the hand has been in between
    # the current time and interval dTime seconds
    def getAveragePosByTime(self, dTime):
        if len(self.positions) == 0:
            return None, 0
        now = time.time()
        # latestTime = self.positions[-1].now
        avgX, avgY, avgZ, num = 0, 0, 0, 0
        for point in self.positions[-1::-1]: # reverse order
            if now - point.header.stamp.secs <= dTime:
                avgX += point.point.x
                avgY += point.point.y
                avgZ += point.point.z
                num += 1
            else:
                break # Assume times are in non-decreasing order
        if num == 0:
            return None, num
        avgX /= num
        avgY /= num
        avgZ /= num
        # self.avgX = avgX
        # self.avgY = avgY
        # self.avgW = avgW
        # self.avgH = avgH
        pointMsg = PointStamped()
        pointMsg.header = self.positions[-1].header
        # pointMsg.header = Header(stamp=rospy.Time.now(), frame_id='/camera_link')
        pointMsg.point = Point()
        # COORDINATE TRANSFORMATION!!!!
        pointMsg.point.x = avgX
        pointMsg.point.y = avgY
        pointMsg.point.z = avgZ
        return pointMsg, num

class HandDetector(object):

    def __init__(self, topic, topicRate, cameraName, handModelPath, maxDx, maxDy, maxDz, timeToDeleteHand, groundDzThreshold,
    avgPosDtime, avgHandXYZDtime, maxIterationsWithNoDifference, differenceThreshold, differenceFactor,
    cascadeScaleFactor, cascadeMinNeighbors, handHeightIntervalDx, handHeightIntervalDy, getDepthAtMidpointOfHand, getAveragePos):
        self.bridge = CvBridge()
        self.fgbg = cv2.BackgroundSubtractorMOG()
        # self.display_pub= rospy.Publisher('/robot/xdisplay',Image, queue_size=10)
        self.hands = [] # list of hand objects
        self.handsLock = threading.Lock()
        self.shouldUpdateBaxter = False
        self.shouldUpdateBaxterLock = threading.Lock()
        self.maxDx = maxDx
        self.maxDy = maxDy
        self.maxDz = maxDz
        self.timeToDeleteHand = timeToDeleteHand
        self.hands_cascade = cv2.CascadeClassifier(handModelPath)
        self.killThreads = False
        self.rgbData = None
        self.rgbDataLock = threading.Lock()
        rgbThread = threading.Thread(target=self.cascadeClassifier)
        rgbThread.daemon = True
        rgbThread.start()
        self.detectedHandsRect = []
        self.detectedHandsRectLock = threading.Lock()
        self.detectedHandsXYZ = []
        self.detectedHandsXYZLock = threading.Lock()
        self.depthData = None
        self.depthDataLock = threading.Lock()
        depthThread = threading.Thread(target=self.getHandDepth)
        depthThread.daemon = True
        depthThread.start()
        # self.handCoord = []
        # self.handCoordLock = threading.Lock()
        classifyHandsThread = threading.Thread(target=self.classifyHands, args=())
        classifyHandsThread.daemon = True
        classifyHandsThread.start()
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.rgbKinectcallback, queue_size=1)
        self.depth_registered_points_sub = rospy.Subscriber("/camera/depth_registered/points",PointCloud2,self.depthKinectcallback)
        self.handPositionPublisher = rospy.Publisher(topic, PointStamped, queue_size=1)
        self.avgX = None
        self.avgY = None
        self.avgZ = None
        self.groundZ = None
        self.dZ = groundDzThreshold # Ignore all points with height +/- dz of groundz
        self.cascadeScaleFactor = cascadeScaleFactor
        self.cascadeMinNeighbors = cascadeMinNeighbors
        avgPosThread = threading.Thread(target=self.getPosOfMostLikelyHand, args=(avgPosDtime,topicRate))
        avgPosThread.daemon = True
        avgPosThread.start()
        self.avgHandXYZDtime = avgHandXYZDtime
        self.threads = [rgbThread, depthThread, avgPosThread, classifyHandsThread]
        self.iterationsWithNoDifference = 0
        self.maxIterationsWithNoDifference = maxIterationsWithNoDifference
        self.differenceThreshold = differenceThreshold
        self.differenceFactor = differenceFactor
        self.handHeightIntervalDx = handHeightIntervalDx
        self.handHeightIntervalDy = handHeightIntervalDy
        self.getDepthAtMidpointOfHand = getDepthAtMidpointOfHand
        self.getAveragePos = getAveragePos

    def rgbKinectcallback(self,data):
        if self.rgbDataLock.acquire(False):
            self.rgbData = data
            self.rgbDataLock.release()

    def cascadeClassifier(self):
        # rate = rospy.Rate(1)
        try:
            cam_info = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo, timeout=None)
            img_proc = PinholeCameraModel()
            img_proc.fromCameraInfo(cam_info)
            img, oldImg = None, None
            while not rospy.is_shutdown() and not self.killThreads:
                self.rgbDataLock.acquire(True)
                if self.rgbData is None:
                    self.rgbDataLock.release()
                    continue
                try:
                    if img is not None:
                        oldImg = np.copy(img)
                        # print("reset oldImg")
                    img = self.bridge.imgmsg_to_cv2(self.rgbData, "bgr8")
                    self.rgbDataLock.release()
                except CvBridgeError as e:
                    self.rgbDataLock.release()
                    print(e)
                    continue
                height, width, channels = img.shape
                # print("share", height, width, channels)
                if oldImg is not None and img is not None:
                    diff = np.nonzero(cv2.subtract(img, oldImg) > self.differenceThreshold)[0].shape
                    print("diff", diff, height*width*channels, height*width*channels*self.differenceFactor)
                    if diff[0] < height*width*channels*self.differenceFactor:
                        self.iterationsWithNoDifference += 1
                        print("diff < threshold, iterations", self.iterationsWithNoDifference)
                        if self.iterationsWithNoDifference > self.maxIterationsWithNoDifference:
                            print("remove hands")
                            self.avgX = None
                            self.avgY = None
                            # self.handCoordLock.acquire()
                            # self.handCoord = []
                            # self.handCoordLock.release()
                            self.handsLock.acquire()
                            self.hands = [] # list of hand objects
                            self.handsLock.release()
                            # reset background
                            self.fgbg = cv2.BackgroundSubtractorMOG()
                            self.iterationsWithNoDifference = 0
                        # else:
                            # rate.sleep()
                            # continue
                    else:
                        self.iterationsWithNoDifference = 0


                fgmask = self.fgbg.apply(img)
                hands = self.hands_cascade.detectMultiScale(fgmask, self.cascadeScaleFactor, self.cascadeMinNeighbors)
                self.detectedHandsRectLock.acquire()
                self.detectedHandsRect = []
                print("Reset detectedHandsRect")
                # self.detectedHandsRectLock.release()
                for (x,y,w,h) in hands:
                    if not (math.isnan(x) or math.isnan(y) or math.isnan(w) or math.isnan(h)):
                        # self.detectedHandsRectLock.acquire()
                        self.detectedHandsRect.append(Rectangle(x, y, w, h, time.time()))
                        print("Added rect to detectedHandsRect")
                        # self.detectedHandsRectLock.release()
                        # self.addToHands(x,y,w,h)
                        # if dx is None and dy is None: # First time this hand was registered
                        #     self.shouldUpdateBaxterLock.acquire()
                        #     self.shouldUpdateBaxter = True
                        #     self.shouldUpdateBaxterLock.release()
                        # print(self.hands)
                        # NOTE (amal): Why I have to reverse X, I have no idea...
                        # xPrime = x
                        # yPrime = y
                        rectColor = (255,0,0)
                        rectThickness = 2
                        cv2.rectangle(fgmask,(x,y),(x+w,y+h), rectColor, rectThickness)
                        # NOTE (amal): Why I have to subtract width for x, I have no idea...
                        # midX, midY = xPrime-w/2, yPrime-h/2
                        # print("actual mid", midX, midY)
                        # (x, y, z) = img_proc.projectPixelTo3dRay((midX, midY))
                        # pointMsg = PointStamped()
                        # pointMsg.header = Header(stamp=rospy.Time.now(), frame_id='/camera_rgb_optical_frame')
                        # pointMsg.point = Point()
                        # # COORDINATE TRANSFORMATION!!!!
                        # pointMsg.point.x = x
                        # pointMsg.point.y = y
                        # pointMsg.point.z = z
                        # self.handPositionPublisher.publish(pointMsg)
                        # self.avgX = x
                        # self.avgY = y
                        # self.avgZ = z
                self.detectedHandsRectLock.release()
                if self.avgX is not None and self.avgY is not None:# and self.avgZ is not None:
                    # pass
                    # print("avgX and Y", self.avgX, self.avgY)
                    # print("hand x and y", x, y)
                    # u, v = img_proc.project3dToPixel((self.avgX, self.avgY, self.avgZ))
                    # print("uv", u, v, "avgXYZ", self.avgX, self.avgY, self.avgZ, "wh", width, height)
                    # print("avgXY", self.avgX, self.avgY)
                    circleRadius  = 30
                    circleColor = (255,0,0)
                    circleThickness = 2
                    cv2.circle(fgmask, (int(self.avgX), int(self.avgY)),circleRadius,circleColor,circleThickness)
                    # cv2.circle(fgmask,(int(width-midX), int(height-midY)),30,(255,0,255), 2)

                cv2.imshow("Image window", fgmask)
                # Number of ms to show the image for
                cv2.waitKey(1)
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return
            # cv2.destroyAllWindows()
            # rospy.sleep(5)


        # img = cv_image
        # fgmask = self.fgbg.apply(img)

        #gray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
        # hands = self.hands_cascade.detectMultiScale(fgmask, 1.3, 5)

        # for (x,y,w,h) in hands:
        #     self.addToHands(x,y,w,h)
        #     # if dx is None and dy is None: # First time this hand was registered
        #     #     self.shouldUpdateBaxterLock.acquire()
        #     #     self.shouldUpdateBaxter = True
        #     #     self.shouldUpdateBaxterLock.release()
        #     print(self.hands)
        #     cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)

        # cv2.imshow("Image window", fgmask)
        # screen_dis =  self.bridge.cv2_to_imgmsg(fgmask, encoding="passthrough")
        # screen_dis =  self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
        # self.display_pub.publish(screen_dis)
        # cv2.waitKey(1)

    def classifyHands(self, hertz=10):
        rate = rospy.Rate(hertz)
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                self.detectedHandsXYZLock.acquire()
                if len(self.detectedHandsXYZ) == 0:
                    self.detectedHandsXYZLock.release()
                    # TODO (amal): should I do rate.sleep() or time.sleep(1/hertz?)
                    # rate.sleep()
                    # print("hand is none")
                    continue
                points = copy.copy(self.detectedHandsXYZ)
                print("Copied detectedHandsXYZ")
                self.detectedHandsXYZLock.release()
                for point in points:
                    self.addToHands(point)
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

    def addToHands(self, point):
        self.handsLock.acquire()
        i = 0
        while True: # I do this to avoid python calculating the len at the beginning and not accounting for changes to the length through this loop
            if i >= len(self.hands):
                break
            hand = self.hands[i]
            recentPoint = hand.getLatestPos()
            # Lazily remove detected hands if they are too old
            if recentPoint is None or recentPoint.header.stamp.secs <= time.time()-self.timeToDeleteHand:
                self.hands.pop(i)
                continue
            # dx, dy, added = hand.addPosition(rect)
            # if added:
            #     self.handsLock.release()
            #     return dx, dy
            if hand.addPosition(point):
                print("added to hands 1")
                self.handsLock.release()
                return
            i += 1
        self.hands.append(Hand(point, self.maxDx, self.maxDy, self.maxDz))
        print("added to hands 2")
        self.handsLock.release()
        return# None, None

    # TODO (amal): what if instead of determining most likely hand by a
    # majority vote, I augment that with how close the hand is to the center
    # of the robot (in this case I would have to change the voting part to
    # after we get the depth for all hands)?  Might help prevent the robot
    # from going to either the robot arm as a hand, or tthe ground as a hand
    # def getMostRecentXYOfMostLikelyHand(self):
    #     self.handsLock.acquire()
    #     if len(self.hands) == 0:
    #         self.handsLock.release()
    #         # print("hand len is 0")
    #         return None
    #     maxNum, maxAvgPos, maxI = 0, None, None
    #     for i in xrange(len(self.hands)):
    #         hand = self.hands[i]
    #         pos, num = hand.getAveragePosByTime(self.avgHandXYZDtime)
    #         # print("abc", pos, num, hand)
    #         if num > maxNum:
    #             maxNum = num
    #             maxAvgPos = pos
    #             maxI = i
    #     self.handsLock.release()
    #     if maxI is None:
    #         return None
    #     mostRecentPos = self.hands[maxI].getLatestPos()
    #     if mostRecentPos is None:
    #         print("mostRecentPos was None!!!!!! :O")
    #         return None
    #     else:
    #         return mostRecentPos

    def depthKinectcallback(self, data):
        if self.depthDataLock.acquire(False):
            #print("release acqurie 1")
            if self.depthData is None: # First timeToSleep
                # Get ground z
                # TODO (amal): chck skipNans!
                maxZ = None
                for handCoord in pc2.read_points(data, field_names=None, skip_nans=True):
                    if not (math.isnan(handCoord[2]) or handCoord[2]==0.0):
                        if maxZ is None or handCoord[2] > maxZ:
                            maxZ = handCoord[2]
                self.groundZ = maxZ
                print("groundZ", maxZ)
            self.depthData = data
            #print("release 4")
            self.depthDataLock.release()

    def getHandDepth(self, hertz=10):
        rate = rospy.Rate(hertz)
        try:
            # cam_info = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo, timeout=None)
            # img_proc = PinholeCameraModel()
            # img_proc.fromCameraInfo(cam_info)
            while not rospy.is_shutdown() and not self.killThreads:
                # TODO (amal): perhaps instead of only popping one, we pop until it is empty?
                self.detectedHandsRectLock.acquire()
                if len(self.detectedHandsRect) == 0:
                    self.detectedHandsRectLock.release()
                    # TODO (amal): should I do rate.sleep() or time.sleep(1/hertz?)
                    # rate.sleep()
                    # print("hand is none")
                    continue
                rects = copy.copy(self.detectedHandsRect)
                print("Copied detectedHandsRect")
                self.detectedHandsRectLock.release()

                self.detectedHandsXYZLock.acquire()
                self.detectedHandsXYZ = []
                print("Reset detectedHandsXYZ")
                # self.detectedHandsXYZLock.release()

                for rect in rects:
                    # rect = self.getMostRecentXYOfMostLikelyHand()
                    # print("recent pos of likely hand is ", rect)
                    # if rect is None:
                    #     time.sleep(1/hertz)
                    #     # print("hand is none")
                    #     continue
                    # print("Most Likely Hand x y", rect)
                    # xPrime = self.kinectRGBWidth-rect.x
                    # yPrime = self.kinectRGBHeight-rect.y
                    # midX = rect.x + rect.w/2
                    # midY = rect.y + rect.h/2
                    # print("calc mid 1", midX, midY)
                    # (x, y, z) = img_proc.projectPixelTo3dRay((midX, midY))
                    # self.avgX = midX
                    # self.avgY = midY
                    # self.avgZ = z
                    # self.avgX = midX
                    # self.avgY = midY
                    self.depthDataLock.acquire(True)
                    #print("release acqurie 2")
                    if self.depthData is None:
                        #print("release 1")
                        self.depthDataLock.release()
                        # print("depth data is None")
                        # TODO (amal): should I do rate.sleep() or time.sleep(1/hertz?)
                        # rate.sleep()
                        continue
                    uvs = []
                    avgX, avgY, avgZ, num = float(0),float(0),None,0
                    if self.getDepthAtMidpointOfHand:
                        midX = rect.x + rect.w/2
                        midY = rect.y + rect.h/2
                        uvs = [(midX, midY)]
                    else:
                        dx = self.handHeightIntervalDx
                        dy = self.handHeightIntervalDy
                        # TODO (amal): change this hardcoded large number!
                        # avgX, avgY, avgZ, num = float(0),float(0),float(0),0
                        for x in xrange(rect.x, rect.x+rect.w, dx):
                            for y in xrange(rect.y, rect.y+rect.h, dy):
                                if (x >= self.depthData.width) or (y >= self.depthData.height):
                                    #print("release 2")
                                    # print("hand coord is out of pix")
                                    continue
                                uvs.append((x,y))
                        # print("uvs", uvs)
                    try:
                        # TODO (amal): play around with this Nan thing!
                        data_out = pc2.read_points(self.depthData, field_names=None, skip_nans=True, uvs=uvs)
                    except e:
                        print(e)
                        self.depthDataLock.release()
                        continue
                    #print("release 3")
                    self.depthDataLock.release()
                    for i in xrange(len(uvs)):
                        try:
                            handCoord = next(data_out)
                        except StopIteration:
                            # print("got nan")
                            break
                            # TODO (amal): what happens when robot hand on top of human hand?
                        if not (math.isnan(handCoord[0]) or handCoord[0]==0.0 or math.isnan(handCoord[1]) or handCoord[1]==0.0 or math.isnan(handCoord[2]) or handCoord[2]==0.0 or handCoord[2] > self.groundZ - self.dZ):
                            # if handCoord[0] < avgX:
                                # avgX = handCoord[0]
                            avgX += handCoord[0]
                            # if avgY is None or handCoord[1] < avgY:
                            #     avgY = handCoord[1] # min not avg
                            avgY += handCoord[1]
                            if avgZ is None or handCoord[2] < avgZ:
                                avgZ = handCoord[2] # min not avg
                            # avgZ += handCoord[2]
                            num += 1
                    if num == 0:
                        # print("got no points with intensity")
                        # TODO (amal): should I do rate.sleep() or time.sleep(1/hertz?)
                        # rate.sleep()
                        continue
                    # print("avg in depth", avgX, avgY, avgZ, num)
                    avgX /= float(num)
                    avgY /= float(num)
                    # avgZ /= float(num)
                    # print("depth", avgX, avgY, avgZ)
                    pointMsg = PointStamped()
                    pointMsg.header = self.depthData.header
                    # pointMsg.header = Header(stamp=rospy.Time.now(), frame_id='/camera_link')
                    pointMsg.point = Point()
                    # COORDINATE TRANSFORMATION!!!!
                    pointMsg.point.x = avgX
                    pointMsg.point.y = avgY
                    pointMsg.point.z = avgZ
                    # self.avgX = avgX
                    # self.avgY = avgY
                    # self.avgZ = avgZ
                    # self.handPositionPublisher.publish(pointMsg)
                    # self.detectedHandsXYZLock.acquire()
                    self.detectedHandsXYZ.append(pointMsg)
                    print("Added point to detectedHandsXYZ")
                self.detectedHandsXYZLock.release()

                # rospy.loginfo("hand coord " + repr(handCoord) + "len "+str(len(self.handCoord)))
                # TODO (amal): should I do rate.sleep() or time.sleep(1/hertz?)
                # rate.sleep()
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

    # Gets the average of the last nTimes handCoords the hand has been in
    # def getAveragePosByNum(self, nTimes):
    #     self.handCoordLock.acquire()
    #     if nTimes <= 0 or len(self.handCoord) == 0:
    #         self.handCoordLock.release()
    #         return None
    #     num = min(nTimes, len(self.handCoord))
    #     avgX, avgY, avgZ = 0, 0, 0
    #     for i in xrange(len(self.handCoord)-num, len(self.handCoord)):
    #         avgX += self.handCoord[i].x
    #         avgY += self.handCoord[i].y
    #         avgZ += self.handCoord[i].z
    #     self.handCoordLock.release()
    #     avgX /= num
    #     avgY /= num
    #     avgZ /= num
    #     self.avgX = avgX
    #     self.avgY = avgY
    #     return Coord(avgX, avgY, avgZ, None), nTimes

    # TODO (amal): what if instead of determining most likely hand by a
    # majority vote, I augment that with how close the hand is to the center
    # of the robot (in this case I would have to change the voting part to
    # after we get the depth for all hands)?  Might help prevent the robot
    # from going to either the robot arm as a hand, or tthe ground as a hand
    # Gets the average of the last handCoord that the hand has been in between
    # the current time and interval dTime seconds
    def getPosOfMostLikelyHand(self, dTime, hertz):
        rate = rospy.Rate(hertz)
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                # rate.sleep()
                print("in getAveragePosByTime")
                self.handsLock.acquire()
                if len(self.hands) == 0:
                    print("len of self.hands is 0", self.hands)
                    self.handsLock.release()
                    # return None, 0
                    # rate.sleep()
                    continue
                maxNum, maxAvgPoint, maxI = 0, None, None
                for i in xrange(len(self.hands)):
                    hand = self.hands[i]
                    pos, num = hand.getAveragePosByTime(self.avgHandXYZDtime)
                    print("got avg pos of hand", pos, num)
                    if num > maxNum:
                        maxNum = num
                        maxAvgPoint = pos
                        maxI = i
                self.handsLock.release()
                if maxNum == 0:
                    # rate.sleep()
                    continue
                if self.getAveragePos:
                    # if maxAvgPoint is None:
                    #     print("maxAvgPoint was None!!!!!! :O")
                    #     # return None, 0
                    #     # rate.sleep()
                    #     continue
                    pointToPublish = maxAvgPoint
                else:
                    pointToPublish = self.hands[maxI].getLatestPos()
                self.handPositionPublisher.publish(pointToPublish)
                print("published pos", maxAvgPoint)
                continue
                # now = time.time()
                # # latestTime = self.handCoord[-1].now
                # avgX, avgY, avgZ, num = 0, 0, 0, 0
                # for coord in self.handCoord[-1::-1]: # reverse order
                #     # print("times", now,  int(coord.header.stamp.to_sec()))
                #     if now - coord.header.stamp.to_sec() <= dTime:
                #         avgX += coord.point.x
                #         avgY += coord.point.y
                #         avgZ += coord.point.z
                #         num += 1
                #     else:
                #         break # Assume times are in non-decreasing order
                # if num == 0:
                #     self.handCoordLock.release()
                #     # print("num is 0", num)
                #     # return None, num
                #     continue
                # avgX /= num
                # avgY /= num
                # avgZ /= num
                # # self.avgX = avgX
                # # self.avgY = avgY
                # # self.avgZ = avgZ
                # print("avgposbytime", avgX, avgY, avgZ)

                # pointMsg = PointStamped()
                # pointMsg.header = self.handCoord[-1].header
                # pointMsg.point = Point()
                # # COORDINATE TRANSFORMATION!!!!
                # pointMsg.point.x = avgX
                # pointMsg.point.y = avgY
                # pointMsg.point.z = avgZ
                # self.handPositionPublisher.publish(pointMsg)
                # self.handCoordLock.release()
                # print("published!")
        except KeyboardInterrupt, rospy.ROSInterruptException:
          return
        # COORDINATE TRANSFORMATION!!!!
        # return pointMsg, num

def main(args):
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-c', '--camera', required=True,
        help="the camera name"
    )
    args = parser.parse_args(rospy.myargv()[1:])
    # We don't want multiple instances of this node running
    rospy.init_node('HandDetector', anonymous=False)
    detector = HandDetector(
        topic=rospy.get_param("reachingHand/topic"),
        topicRate=rospy.get_param("reachingHand/HandDetector/topicRate"),
        cameraName=args.camera,
        handModelPath=rospy.get_param("reachingHand/HandDetector/handModelPath"),
        maxDx=rospy.get_param("reachingHand/HandDetector/maxAllowedHandMotion/dx"),
        maxDy=rospy.get_param("reachingHand/HandDetector/maxAllowedHandMotion/dy"),
        maxDz=rospy.get_param("reachingHand/HandDetector/maxAllowedHandMotion/dz"),
        timeToDeleteHand=rospy.get_param("reachingHand/HandDetector/timeToDeleteHand"),
        groundDzThreshold=rospy.get_param("reachingHand/HandDetector/groundDzThreshold"),
        avgPosDtime=rospy.get_param("reachingHand/HandDetector/avgPosDtime"),
        avgHandXYZDtime=rospy.get_param("reachingHand/HandDetector/avgHandXYZDtime"),
        maxIterationsWithNoDifference=rospy.get_param("reachingHand/HandDetector/imageDifferenceParams/maxIterations"),
        differenceThreshold=rospy.get_param("reachingHand/HandDetector/imageDifferenceParams/differenceThreshold"),
        differenceFactor=rospy.get_param("reachingHand/HandDetector/imageDifferenceParams/differenceFactor"),
        cascadeScaleFactor=rospy.get_param("reachingHand/HandDetector/cascadeClassifierParams/scale"),
        cascadeMinNeighbors=rospy.get_param("reachingHand/HandDetector/cascadeClassifierParams/minNeighbors"),
        handHeightIntervalDx=rospy.get_param("reachingHand/HandDetector/handHeightInterval/dx"),
        handHeightIntervalDy=rospy.get_param("reachingHand/HandDetector/handHeightInterval/dy"),
        getDepthAtMidpointOfHand=rospy.get_param("reachingHand/HandDetector/getDepthAtMidpointOfHand"),
        getAveragePos=rospy.get_param("reachingHand/HandDetector/getAveragePos"),
    )
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
