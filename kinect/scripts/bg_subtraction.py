#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import roslib
from roslib import message
roslib.load_manifest('test_cam')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import baxter_interface
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
import time
import threading
from baxter_interface import settings

class Coord(object):
    def __init__(self, x, y, z, now):
        self.x = x
        self.y = y
        self.z = z
        self.now = now

class Rectangle(object):
    def __init__(self, x, y, w, h, now):
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
        latestTime = self.positions[-1].now
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

class ImageConverter(object):

    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.rgbKinectcallback, queue_size=1)
        self.depth_registered_points_sub = rospy.Subscriber("/camera/depth_registered/points",PointCloud2,self.depthKinectcallback)
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
        self.rgbData = None
        self.rgbDataLock = threading.Lock()
        self.rgbThread = threading.Thread(target=self.cascadeClassifier)
        self.rgbThread.daemon = True
        self.rgbThread.start()
        self.depthData = None
        self.depthDataLock = threading.Lock()
        self.depthThread = threading.Thread(target=self.getHandDepth)
        self.depthThread.daemon = True
        self.depthThread.start()
        self.handCoord = []
        self.handCoordLock = threading.Lock()

        # Get the transform between the camera and base at the beginning, and assume it doesn't change

    def rgbKinectcallback(self,data):
        if self.rgbDataLock.acquire(False):
            self.rgbData = data
            self.rgbDataLock.release()

    def cascadeClassifier(self):
        while True:
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

            for (x,y,w,h) in hands:
                self.addToHands(x,y,w,h)
                # if dx is None and dy is None: # First time this hand was registered
                #     self.shouldUpdateBaxterLock.acquire()
                #     self.shouldUpdateBaxter = True
                #     self.shouldUpdateBaxterLock.release()
                # print(self.hands)
                cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)

            cv2.imshow("Image window", fgmask)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # time.sleep(5)


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

    def getAvgXYOfMostLikelyHand(self):
        dTime = 2
        self.handsLock.acquire()
        if len(self.hands) == 0:
            self.handsLock.release()
            # print("hand len is 0")
            return None
        maxNum, maxAvgPos = 0, None
        for hand in self.hands:
            pos, num = hand.getAveragePosByTime(dTime)
            # print(pos, num, hand)
            if num > maxNum:
                maxNum = num
                maxAvgPos = pos
        self.handsLock.release()
        return maxAvgPos

    def depthKinectcallback(self, data):
        if self.depthDataLock.acquire(False):
            self.depthData = data
            self.depthDataLock.release()

    def getHandDepth(self):
        while True:
            rect = self.getAvgXYOfMostLikelyHand()
            if rect is None:
                # print("hand is none")
                continue
            # print("Most Likely Hand x y", rect)
            midX = rect.x + rect.w/2
            midY = rect.y + rect.h/2
            self.depthDataLock.acquire(True)
            if self.depthData is None:
                self.depthDataLock.release()
                # print("depth data is None")
                continue
            if (midX >= self.depthData.height) or (midY >= self.depthData.width):
                self.depthDataLock.release()
                # print("hand coord is out of pix")
                continue
            data_out = pc2.read_points(self.depthData, field_names=None, skip_nans=False, uvs=[(midX, midY)])
            self.depthDataLock.release()
            handCoord = next(data_out)
            self.handCoordLock.acquire()
            self.handCoord.append(Coord(handCoord[0], handCoord[1], handCoord[2], time.time()))
            self.handCoordLock.release()
            rospy.loginfo("hand coord " + repr(handCoord))

    # Gets the average of the last nTimes handCoords the hand has been in
    def getAveragePosByNum(self, nTimes):
        self.handCoordLock.acquire()
        if nTimes <= 0 or len(self.handCoord) == 0:
            self.handCoordLock.release()
            return None
        num = min(nTimes, len(self.handCoord))
        avgX, avgY, avgZ = 0, 0, 0
        for i in xrange(len(self.handCoord)-num, len(self.handCoord)):
            avgX += self.handCoord[i].x
            avgY += self.handCoord[i].y
            avgZ += self.handCoord[i].z
        self.handCoordLock.release()
        avgX /= num
        avgY /= num
        avgZ /= num
        return Coord(avgX, avgY, avgZ, None), nTimes

    # Gets the average of the last handCoord that the hand has been in between
    # the current time and interval dTime seconds
    def getAveragePosByTime(self, dTime):
        self.handCoordLock.acquire()
        if len(self.handCoord) == 0:
            self.handCoordLock.release()
            return None, 0
        now = time.time()
        latestTime = self.positions[-1].now
        avgX, avgY, avgZ, num = 0, 0, 0, 0
        for coord in self.positions[-1::-1]: # reverse order
            if now - coord.now <= dTime:
                avgX += rect.x
                avgY += rect.y
                avgZ += rect.z
                num += 1
            else:
                break # Assume times are in non-decreasing order
        if num == 0:
            return None, num
        avgX /= num
        avgY /= num
        avgZ /= num
        return Coord(avgX, avgY, avgZ, None), num

class BaxterMovementController(object):
    def __init__(self, limb, speedRatio=0.3):
        self.limb = limb
        self.speedRatio = 0.3
        self.pub_joint_cmd = rospy.Publisher(
            '/robot/limb/' + limb + '/joint_command',
            JointCommand,
            tcp_nodelay=True,
            queue_size=1)
        self.pub_speed_ratio = rospy.Publisher(
            '/robot/limb/' + limb + '/set_speed_ratio',
            Float64,
            latch=True,
            queue_size=10)
        self.jointAngles = dict()
        self.jointAnglesLock = threading.Lock()
        self.joint_state_sub = rospy.Subscriber(
            joint_state_topic,
            JointState,
            self.onJointStates,
            queue_size=1,
            tcp_nodelay=True)
        # transform = TransformListener()
        # TODO: if multiple users, how know which hand to follow?
        self.pub_speed_ratio.publish(Float64(self.speedRatio)) # Limit arm speed
        self.iksvcString = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        self.shouldUpdateBaxterTarget = False
        self.shouldUpdateBaxterTargetLock = threading.Lock()
        self.targetPosition = None
        self.targetPositionLock = threading.Lock()

    def onJointStates(self):
        for i, name in enumerate(msg.name, msg):
            if self.limb in name:
                self.jointAnglesLock.acquire()
                self.jointAngle[name] = msg.position[i]
                self.jointAnglesLock.release()

    def moveArm(self, threshold=settings.JOINT_ANGLE_TOLERANCE, timeToSleep=2):
        def genf(joint, angle):
            def jointDiff():
                self.jointAnglesLock.acquire()
                if len(self.jointAngle) == 0:
                    retVal = 0
                else:
                    retVal = abs(angle - self.jointAngle[joint])
                self.jointAnglesLock.release()
                return retVal
            return jointDiff

        diffs = [genf(j, a) for j, a in positions.items() if j in self.jointAngle]

        reachedTarget = True
        commandMsg = JointCommand()
        while True:
            if reachedTarget:
                self.shouldUpdateBaxterTargetLock.acquire()
                if not self.shouldUpdateBaxterTarget:
                    self.shouldUpdateBaxterTargetLock.release()
                    time.sleep(timeToSleep)
                    continue
                self.shouldUpdateBaxterTargetLock.release()
            # self.targetPositionLock.acquire()
            # positions = self.targetPosition
            # self.targetPositionLock.release()

            def filtered_cmd(positions):
                # First Order Filter - 0.2 Hz Cutoff
                for joint in positions.keys():
                    cmd[joint] = 0.012488 * self.targetPosition[joint] + 0.98751 * cmd[joint] # Get most up to date value
                return cmd

            def loopGuard():
                if rospy.is_shutdown():
                    return False
                for diff in diffs:
                    if diff() > threshold:
                        return True
                return False

            while loopGuard() :
                commandMsg.names = positions.keys()
                commandMsg.command = positions.values()
                commandMsg.mode = JointCommand.RAW_POSITION_MODE
                self.pub_joint_cmd.publish(commandMsg)
            reachedTarget = True
            # ikreq = SolvePositionIKRequest()
            # pose = PoseStamped(
            #     header=hdr,
            #     pose=Pose(
            #         position=Point(
            #             x=x,
            #             y=y,
            #             z=x,
            #         ),
            #         orientation=Quaternion(
            #             x=1.0,
            #             y=0.0,
            #             z=0.0,
            #             w=0.0,
            #         ),
            #     ),
            # )
            # ikreq.pose_stamp.append(pose)
            # try:
            #     rospy.wait_for_service(self.iksvcString, 5.0)
            #     resp = self.iksvc(ikreq)
            # except (rospy.ServiceException, rospy.ROSException), e:
            #     rospy.logerr("IK service call failed: %s" % (e,))
            #     # TODO (amal): check error handling!!!
            #     continue







def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
