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
    def __init__(self, rect, deltaX=100, deltaY=100):
        if type(rect) != Rectangle:
            raise TypeException("first argument to hand must be of type rectangle")
        self.positions = [rect]
        self.deltaX = deltaX
        self.deltaY = deltaY
        # self.avgX, self.avgY, self.avgW, self.avgH = None, None, None, None

    def __str__(self):
        string = "[\n"
        for rect in self.positions:
            string += "     "+str(rect)+"\n"
        return string+"]"

    def __repr__(self):
        return self. __str__()

    def couldHaveMovedHere(self, rect):
        return (abs(rect.x-self.positions[-1].x) < self.deltaX and
        abs(rect.y-self.positions[-1].y) < self.deltaY)

    # returns whether the rect was appended (if the hand could viably move there).
    # if the average rect was previously calculated, it also returns
    def addPosition(self, rect):
        if rect.now is None:
            # return None, None, False
            return False
        if self.couldHaveMovedHere(rect):
            self.positions.append(rect)
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

    def getLastestPos(self):
        if len(self.positions) == 0:
            return None
        return self.positions[-1]

    # Gets the average of the last nTimes positions the hand has been in
    def getAveragePosByNum(self, nTimes):
        if nTimes <= 0 or len(self.positions) == 0:
            return None
        num = min(nTimes, len(self.positions))
        avgX, avgY, avgW, avgH = 0, 0, 0, 0
        for i in xrange(len(self.positions)-num, len(self.positions)):
            avgX += self.positions[i].x
            avgY += self.positions[i].y
            avgW += self.positions[i].w
            avgH += self.positions[i].h
        avgX /= num
        avgY /= num
        avgW /= num
        avgH /= num
        # self.avgX = avgX
        # self.avgY = avgY
        # self.avgW = avgW
        # self.avgH = avgH
        return Rectangle(avgX, avgY, avgW, avgH, None), nTimes

    # Gets the average of the last positions that the hand has been in between
    # the current time and interval dTime seconds
    def getAveragePosByTime(self, dTime):
        if len(self.positions) == 0:
            return None, 0
        now = time.time()
        # latestTime = self.positions[-1].now
        avgX, avgY, avgW, avgH, num = 0, 0, 0, 0, 0
        for rect in self.positions[-1::-1]: # reverse order
            if now - rect.now <= dTime:
                avgX += rect.x
                avgY += rect.y
                avgW += rect.w
                avgH += rect.h
                num += 1
            else:
                break # Assume times are in non-decreasing order
        if num == 0:
            return None, num
        avgX /= num
        avgY /= num
        avgW /= num
        avgH /= num
        # self.avgX = avgX
        # self.avgY = avgY
        # self.avgW = avgW
        # self.avgH = avgH
        return Rectangle(avgX, avgY, avgW, avgH, None), num

class HandDetector(object):

    def __init__(self, topic):
        self.bridge = CvBridge()
        self.fgbg = cv2.BackgroundSubtractorMOG()
        # self.display_pub= rospy.Publisher('/robot/xdisplay',Image, queue_size=10)
        self.hands = [] # list of hand objects
        self.handsLock = threading.Lock()
        self.shouldUpdateBaxter = False
        self.shouldUpdateBaxterLock = threading.Lock()
        self.maxDx = 100
        self.maxDy = 100
        self.maxDz = 100
        self.hands_cascade = cv2.CascadeClassifier('/home/amal/baxter_ws/src/Lab-Baxter-Reaching-Hand/kinect/haarcascade_hand.xml')
        self.killThreads = False
        self.rgbData = None
        self.rgbDataLock = threading.Lock()
        rgbThread = threading.Thread(target=self.cascadeClassifier)
        rgbThread.daemon = True
        rgbThread.start()
        self.depthData = None
        self.depthDataLock = threading.Lock()
        depthThread = threading.Thread(target=self.getHandDepth)
        depthThread.daemon = True
        depthThread.start()
        self.handCoord = []
        self.handCoordLock = threading.Lock()
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.rgbKinectcallback, queue_size=1)
        self.depth_registered_points_sub = rospy.Subscriber("/camera/depth_registered/points",PointCloud2,self.depthKinectcallback)
        self.handPositionPublisher = rospy.Publisher(topic, PointStamped, queue_size=1)
        self.avgX = None
        self.avgY = None
        self.avgZ = None
        # TODO (amal): make these adjustable
        self.groundZ = None
        self.dZ = 0.5 # Ignore all points with height +/- dz of groundz
        # self.kinectRGBHeight = 480
        # self.kinectRGBWidth = 640
        dTime = 0.5
        avgPosThread = threading.Thread(target=self.getAveragePosByTime, args=(dTime,))
        avgPosThread.daemon = True
        avgPosThread.start()
        self.threads = [rgbThread, depthThread, avgPosThread]

        # Get the transform between the camera and base at the beginning, and assume it doesn't change

    def rgbKinectcallback(self,data):
        if self.rgbDataLock.acquire(False):
            self.rgbData = data
            self.rgbDataLock.release()

    def cascadeClassifier(self):
        try:
            cam_info = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo, timeout=None)
            img_proc = PinholeCameraModel()
            img_proc.fromCameraInfo(cam_info)
            while not rospy.is_shutdown() and not self.killThreads:
                self.rgbDataLock.acquire(True)
                if self.rgbData is None:
                    self.rgbDataLock.release()
                    continue
                try:
                    img = self.bridge.imgmsg_to_cv2(self.rgbData, "bgr8")
                    self.rgbDataLock.release()
                except CvBridgeError as e:
                    self.rgbDataLock.release()
                    print(e)
                    continue

                fgmask = self.fgbg.apply(img)
                hands = self.hands_cascade.detectMultiScale(fgmask, 1.3, 5)
                height, width, channels = img.shape
                for (x,y,w,h) in hands:
                    if not (math.isnan(x) or math.isnan(y) or math.isnan(w) or math.isnan(h)):
                        self.addToHands(x,y,w,h)
                        # if dx is None and dy is None: # First time this hand was registered
                        #     self.shouldUpdateBaxterLock.acquire()
                        #     self.shouldUpdateBaxter = True
                        #     self.shouldUpdateBaxterLock.release()
                        # print(self.hands)
                        # NOTE (amal): Why I have to reverse X, I have no idea...
                        # xPrime = x
                        # yPrime = y
                        cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)
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
                if self.avgX is not None and self.avgY is not None:# and self.avgZ is not None:
                    # pass
                    # print("avgX and Y", self.avgX, self.avgY)
                    # print("hand x and y", x, y)
                    # u, v = img_proc.project3dToPixel((self.avgX, self.avgY, self.avgZ))
                    # print("uv", u, v, "avgXYZ", self.avgX, self.avgY, self.avgZ, "wh", width, height)
                    # print("avgXY", self.avgX, self.avgY)
                    cv2.circle(fgmask, (int(self.avgX), int(self.avgY)),30,(255,255,255), 2)
                    # cv2.circle(fgmask,(int(width-midX), int(height-midY)),30,(255,0,255), 2)


                cv2.imshow("Image window", fgmask)
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

    def addToHands(self, x, y, w, h):
        self.handsLock.acquire()
        rect = Rectangle(x, y, w, h, time.time())
        for hand in self.hands:
            # dx, dy, added = hand.addPosition(rect)
            # if added:
            #     self.handsLock.release()
            #     return dx, dy
            if hand.addPosition(rect):
                self.handsLock.release()
                return
        self.hands.append(Hand(rect))
        self.handsLock.release()
        return# None, None

    def getMostRecentXYOfMostLikelyHand(self):
        dTime = 0.5
        self.handsLock.acquire()
        if len(self.hands) == 0:
            self.handsLock.release()
            # print("hand len is 0")
            return None
        maxNum, maxAvgPos, maxI = 0, None, None
        for i in xrange(len(self.hands)):
            hand = self.hands[i]
            pos, num = hand.getAveragePosByTime(dTime)
            # print("abc", pos, num, hand)
            if num > maxNum:
                maxNum = num
                maxAvgPos = pos
                maxI = i
        self.handsLock.release()
        if maxI is None:
            return None
        mostRecentPos = self.hands[maxI].getLastestPos()
        if mostRecentPos is None:
            print("mostRecentPos was None!!!!!! :O")
            return None
        else:
            return mostRecentPos

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
                rect = self.getMostRecentXYOfMostLikelyHand()
                # print("recent pos of likely hand is ", rect)
                if rect is None:
                    # print("hand is none")
                    continue
                # print("Most Likely Hand x y", rect)
                # xPrime = self.kinectRGBWidth-rect.x
                # yPrime = self.kinectRGBHeight-rect.y
                midX = rect.x + rect.w/2
                midY = rect.y + rect.h/2
                # print("calc mid 1", midX, midY)
                # (x, y, z) = img_proc.projectPixelTo3dRay((midX, midY))
                self.avgX = midX
                self.avgY = midY
                # self.avgZ = z
                # self.avgX = midX
                # self.avgY = midY
                self.depthDataLock.acquire(True)
                #print("release acqurie 2")
                if self.depthData is None:
                    #print("release 1")
                    self.depthDataLock.release()
                    # print("depth data is None")
                    continue
                uvs = []
                # uvs = [(midX, midY)]
                dx = 10
                dy = 10
                # TODO (amal): change this hardcoded large number!
                # avgX, avgY, avgZ, num = float(0),float(0),float(0),0
                avgX, avgY, avgZ, num = float(0),float(0),None,0
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
                self.handCoordLock.acquire()
                self.handCoord.append(pointMsg)
                self.handCoordLock.release()

                # rospy.loginfo("hand coord " + repr(handCoord) + "len "+str(len(self.handCoord)))
                rate.sleep()
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

    # Gets the average of the last handCoord that the hand has been in between
    # the current time and interval dTime seconds
    def getAveragePosByTime(self, dTime):
        hertz = 20
        rate = rospy.Rate(hertz)
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                rate.sleep()
                # print("in getAveragePosByTime")
                self.handCoordLock.acquire()
                if len(self.handCoord) == 0:
                    # print("len of self.handCoord is 0", self.handCoord)
                    self.handCoordLock.release()
                    # return None, 0
                    continue
                now = time.time()
                # latestTime = self.handCoord[-1].now
                avgX, avgY, avgZ, num = 0, 0, 0, 0
                for coord in self.handCoord[-1::-1]: # reverse order
                    # print("times", now,  int(coord.header.stamp.to_sec()))
                    if now - coord.header.stamp.to_sec() <= dTime:
                        avgX += coord.point.x
                        avgY += coord.point.y
                        avgZ += coord.point.z
                        num += 1
                    else:
                        break # Assume times are in non-decreasing order
                if num == 0:
                    self.handCoordLock.release()
                    # print("num is 0", num)
                    # return None, num
                    continue
                avgX /= num
                avgY /= num
                avgZ /= num
                # self.avgX = avgX
                # self.avgY = avgY
                # self.avgZ = avgZ
                print("avgposbytime", avgX, avgY, avgZ)
                pointMsg = PointStamped()
                pointMsg.header = self.handCoord[-1].header
                pointMsg.point = Point()
                # COORDINATE TRANSFORMATION!!!!
                pointMsg.point.x = avgX
                pointMsg.point.y = avgY
                pointMsg.point.z = avgZ
                self.handPositionPublisher.publish(pointMsg)
                self.handCoordLock.release()
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
        '-t', '--topic', required=True,
        help="the limb to publish hand points to"
    )
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('HandDetector', anonymous=True)
    ic = HandDetector(args.topic)
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
