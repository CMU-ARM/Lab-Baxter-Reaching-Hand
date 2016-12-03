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
from geometry_msgs.msg import PoseStamped
import time
import threading
import struct
import argparse
from tf import TransformListener
import tf
from baxter_interface import settings
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
import actionlib
from lab_reaching_hand_kinect.msg import (
    ReachingHandFeedback,
    ReachingHandResult,
    ReachingHandAction,
)
# import roslaunch
from HandDetector import HandDetector
# TODO (amal): clean up these imports!!!!

class MovementController(object):
    def __init__(self, limb, topic,camera,speedRatio, jointThresholdEnd,
        jointThresholdWarning, updateQueryRate, jointFilteringFactorFar,
        jointFilteringFactorClose, orientationX, orientationY, orientationZ,
        orientationW, timeToWait, name):
        # Set Confiuration Variables
        self.limb = limb
        self.topic=topic
        self.camera = camera
        self.speedRatio=speedRatio
        self.jointThresholdEnd=jointThresholdEnd
        self.jointThresholdWarning=jointThresholdWarning
        self.updateQueryRate=updateQueryRate
        self.jointFilteringFactorFar=jointFilteringFactorFar
        self.jointFilteringFactorClose=jointFilteringFactorClose
        self.orientationX=orientationX
        self.orientationY=orientationY
        self.orientationZ=orientationZ
        self.orientationW=orientationW
        self.timeToWait=timeToWait
        # Set variables that are static across executions of the actionServer
        self.handDetector = self.createHandDetector()
        self.pub_joint_cmd = rospy.Publisher(
            '/robot/limb/' + self.limb + '/joint_command',
            JointCommand,
            tcp_nodelay=True,
            queue_size=1)
        self.pub_speed_ratio = rospy.Publisher(
            '/robot/limb/' + self.limb + '/set_speed_ratio',
            Float64,
            latch=True,
            queue_size=1)
        self.pub_speed_ratio.publish(Float64(speedRatio)) # Limit arm speed
        self.jointAngles = dict()
        self.jointAnglesLock = threading.Lock()
        joint_state_topic = 'robot/joint_states'
        self.joint_state_sub = rospy.Subscriber(
            joint_state_topic,
            JointState,
            self.onJointStates,
            queue_size=1,
            tcp_nodelay=True)
        self.iksvcStringLimb = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self.iksvcLimb = rospy.ServiceProxy(self.iksvcStringLimb, SolvePositionIK)
        # Set Action Server-Related Variables
        self.feedback = ReachingHandFeedback()
        self.result   = ReachingHandResult()
        self.actionServer = actionlib.SimpleActionServer(name, ReachingHandAction, execute_cb=self.execute_cb, auto_start = False)
        self.actionServer.start()

    def createHandDetector(self):
        detector = HandDetector(
            topic=rospy.get_param("reachingHand/topic"),
            topicRate=rospy.get_param("reachingHand/HandDetector/topicRate"),
            cameraName=self.camera,
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
        return detector

    def execute_cb(self, goal):
        # Launch the HandDetector
        # package = 'lab_reaching_hand_kinect'
        # executable = 'HandDetector.py'
        # node = roslaunch.core.Node(package, executable)
        # launch = roslaunch.scriptapi.ROSLaunch()
        # launch.start()
        # handDetectorProcess = launch.launch(node)
        self.handDetector.start()

        # Subscribe to necessary topics and create necessary background threads
        self.targetPose = None
        self.shouldUpdateBaxterTarget = False
        self.shouldUpdateBaxterTargetLock = threading.Lock()
        self.targetPosition = dict()
        self.targetPoint = None
        self.killThreads = False
        self.targetPositionLock = threading.Lock()
        # moveArmThread = threading.Thread(target=self.moveArm, args=(self.jointThresholdEnd, self.jointThresholdWarning, self.updateQueryRate, self.jointFilteringFactorFar, self.jointFilteringFactorClose))
        # moveArmThread.start()
        self.targetEndEffectorPointTopic = rospy.Subscriber(self.topic, PointStamped, self.targetEndEffectorPointCallback, queue_size=1)
        self.targetEndEffectorPoint = None
        self.targetEndEffectorPointLock = threading.Lock()
        endEffectorTopic = "robot/limb/"+self.limb+"/endpoint_state"
        self.actualEndEffectorPoseTopic = rospy.Subscriber(endEffectorTopic, EndpointState, self.actualEndEffectorPoseCallback, queue_size=1)
        self.actualEndEffectorPose = None
        self.actualEndEffectorPoseLock = threading.Lock()
        endEffectorThread = threading.Thread(target=self.setTargetEndEffectorPosition, args=(self.orientationX, self.orientationY, self.orientationZ, self.orientationW))
        endEffectorThread.daemon = True
        endEffectorThread.start()
        # self.threads = [self.endEffectorThread]
        # Move the arm
        self.moveArm(self.jointThresholdEnd, self.jointThresholdWarning, self.updateQueryRate, self.jointFilteringFactorFar, self.jointFilteringFactorClose)
        # Kill threads, unregister topics, and kill the HandDetector Node
        self.killThreads = True
        self.targetEndEffectorPointTopic.unregister()
        self.actualEndEffectorPoseTopic.unregister()
        # handDetectorProcess.stop()
        self.handDetector.stop()


    def onJointStates(self, msg):
        for i, name in enumerate(msg.name):
            # print("onJointStates", i, name)
            if self.limb in name:
                self.jointAnglesLock.acquire()
                self.jointAngles[name] = msg.position[i]
                self.jointAnglesLock.release()

    def actualEndEffectorPoseCallback(self, state):
        if self.actualEndEffectorPoseLock.acquire(False):
            self.actualEndEffectorPose = state.pose
            self.actualEndEffectorPoseLock.release()

    def moveArm(self, jointThresholdEnd, jointThresholdWarning, updateQueryRate,
        jointFilteringFactorFar, jointFilteringFactorClose):
        rate = rospy.Rate(updateQueryRate)
        print("moveArm")
        def genf(targetPoint, targetOrientation):
        # def genf(joint, angle):
            # def jointDiff():
            #     self.jointAnglesLock.acquire()
            #     if len(self.jointAngles) == 0:
            #         retVal = 0
            #     else:
            #         retVal = abs(angle - self.jointAngles[joint])
            #         print(joint, retVal)
            #     self.jointAnglesLock.release()
            #     return retVal
            def endEffectorDiff():
                self.actualEndEffectorPoseLock.acquire()
                if self.actualEndEffectorPose is None:
                    retValPos = 0
                    retValOrien = 0
                else:
                    retValPos = math.sqrt((self.actualEndEffectorPose.position.x-targetPoint.x)**2+
                                          (self.actualEndEffectorPose.position.y-targetPoint.y)**2+
                                          (self.actualEndEffectorPose.position.z-targetPoint.z)**2)
                    retValOrien = math.sqrt((self.actualEndEffectorPose.orientation.x-targetOrientation.x)**2+
                                            (self.actualEndEffectorPose.orientation.y-targetOrientation.y)**2+
                                            (self.actualEndEffectorPose.orientation.z-targetOrientation.z)**2+
                                            (self.actualEndEffectorPose.orientation.w-targetOrientation.w)**2)
                    # print(joint, retVal)
                self.actualEndEffectorPoseLock.release()
                return retValPos, retValOrien
            return endEffectorDiff

        # Wait until we have read the joint state at least once
        # TODO (amal): do something better than a spin lock
        try:
            while not rospy.is_shutdown() and (len(self.jointAngles) == 0 or len(self.targetPosition) == 0 or self.actualEndEffectorPose is None) and not self.killThreads:
                if self.actionServer.is_preempt_requested():
                    self.preempt()
                    return
                rate.sleep()
                continue
        except KeyboardInterrupt, rospy.ROSInterruptException:
            self.abort()
            return

        # TODO (amal): do i have a race condition here from not locking targetPose?
        # def diffs():
        #     # return genf(self.targetPose.position, self.targetPose.orientation)
        #     return [genf(j, a) for j, a in self.targetPosition.items() if j in self.jointAngles]

        reachedTarget = True
        commandMsg = JointCommand()
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                if self.actionServer.is_preempt_requested():
                    self.preempt()
                    return
                if reachedTarget:
                    self.shouldUpdateBaxterTargetLock.acquire()
                    if not self.shouldUpdateBaxterTarget:
                        self.shouldUpdateBaxterTargetLock.release()
                        rate.sleep()
                        continue
                    self.shouldUpdateBaxterTargetLock.release()
                # self.targetPositionLock.acquire()
                # positions = self.targetPosition
                # self.targetPositionLock.release()

                # TODO (amal): look into exponential moving average
                def filtered_cmd():
                    # print(self.targetPosition, self.jointAngles)
                    if not self.closeToTarget:
                        self.targetPositionLock.acquire()
                    retPositions = dict()
                    # First Order Filter - ????? Hz Cutoff
                    # factor = 0.012488
                    factor =jointFilteringFactorFar
                    if self.closeToTarget:
                        factor = jointFilteringFactorClose
                    for joint in  self.targetPosition.keys():
                        if self.currentlyMovingTowardsPoint is None or not self.closeToTarget:
                            self.currentlyMovingTowardsPoint = self.targetPosition
                        retPositions[joint] = factor * self.currentlyMovingTowardsPoint[joint] + (1-factor) * self.jointAngles[joint] # Get most up to date value
                    if not self.closeToTarget:
                        self.targetPositionLock.release()
                    return retPositions

                self.closeToTarget = False
                self.currentlyMovingTowardsPoint = None
                def loopGuard():
                    if rospy.is_shutdown() or self.killThreads:
                        return False
                    allDiffsWithinWarning = True
                    # TODO (amal): is this a race because I don't lock targetPose?
                    positionDiff, orientationDiff = genf(self.targetPose.pose.position, self.targetPose.pose.orientation)()
                    diff = positionDiff+orientationDiff
                    self.feedback.distance = float(diff)
                    self.actionServer.publish_feedback(self.feedback)
                    if diff > jointThresholdWarning:
                        allDiffsWithinWarning = False
                    # for diff in diffs():
                    #     if diff() > jointThresholdWarning:
                    #         allDiffsWithinWarning = False
                    if allDiffsWithinWarning and not self.closeToTarget:
                        print("WITHIN WARNING!")
                        # Don't let the target point get changed from now till the hand reaches its target
                        self.targetPositionLock.acquire()
                        print("acquired target position lock")
                        self.closeToTarget = True
                    positionDiff, orientationDiff = genf(self.targetPose.pose.position, self.targetPose.pose.orientation)()
                    if positionDiff+orientationDiff > jointThresholdEnd:
                        print("outside of threshold")
                        return True
                    # for diff in diffs():
                    #     if diff() > jointThresholdEnd:
                    #         print("outside of threshold")
                    #         return True
                    return False
                try:
                    # print("before loop guard")
                    while loopGuard():
                        # print("loop guard true")
                        position = filtered_cmd()
                        commandMsg.names = position.keys()
                        commandMsg.command = position.values()
                        # commandMsg.command = self.targetPosition.values()
                        # print(self.jointAngles, self.targetPosition, position)
                        commandMsg.mode = JointCommand.POSITION_MODE
                        self.pub_joint_cmd.publish(commandMsg)
                        print("pre end movememnt", self.closeToTarget)
                except KeyboardInterrupt, rospy.ROSInterruptException:
                    if self.closeToTarget:
                        self.targetPositionLock.release()
                        self.closeToTarget = False
                    self.abort()
                    return
                print("ended movememnt", self.closeToTarget)
                if self.closeToTarget:
                    self.targetPositionLock.release()
                    self.closeToTarget = False
                reachedTarget = True
                # for diff in diffs():
                #     if diff() > threshold:
                #         continue
                self.shouldUpdateBaxterTargetLock.acquire()
                self.shouldUpdateBaxterTarget = False
                print("DONE DONE DONE!!!")
                self.shouldUpdateBaxterTargetLock.release()
                self.success()
                return
        except KeyboardInterrupt, rospy.ROSInterruptException:
            self.abort()
            return

    def abort(self):
        print("abort")
        self.actionServer.set_aborted()
        # self.killThreads = True
        # self.targetEndEffectorPointTopic.unregister()
        # self.actualEndEffectorPoseTopic.unregister()
        return

    def success(self):
        print("success")
        self.actionServer.set_succeeded()
        # TODO(amal): if this does not succeed in killing threads, explicitly kill by self.threads list
        return

    def preempt(self):
        print("preempt")
        self.actionServer.set_preempted()
        # self.killThreads = True
        # self.targetEndEffectorPointTopic.unregister()
        # self.actualEndEffectorPoseTopic.unregister()
        return

    def targetEndEffectorPointCallback(self, point):
        if self.targetEndEffectorPointLock.acquire(False):
            # print("got point")
            self.targetEndEffectorPoint = point
            self.targetEndEffectorPointLock.release()

    def setTargetEndEffectorPosition(self, orientationX, orientationY, orientationZ, orientationW):
        listener = TransformListener()
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                # print("setTargetEndEffectorPosition")
                self.targetEndEffectorPointLock.acquire(True)
                # print(self.targetEndEffectorPoint)
                if self.targetEndEffectorPoint is None:
                    self.targetEndEffectorPointLock.release()
                    continue
                try:
                    print("waiting for transofrm from /base to", self.targetEndEffectorPoint.header.frame_id)
                    listener.waitForTransform("/base", self.targetEndEffectorPoint.header.frame_id, rospy.Time(), rospy.Duration(secs=self.timeToWait))
                    transformedPoint = listener.transformPoint("/base", self.targetEndEffectorPoint)
                except (tf.ConnectivityException, tf.ExtrapolationException, tf.Exception) as e:
                    print("TF Failure", e)
                    self.targetEndEffectorPointLock.release()
                    continue
                # If the target point is at similar x and y, but the z has jumped, assume we are detecting the robot hand
                # dx = 0.05
                # dy = 0.05
                # dz = 0.1
                # if self.targetPoint is not None and abs(transformedPoint.point.x - self.targetPoint.point.x) < dx and abs(transformedPoint.point.y - self.targetPoint.point.y) < dy and abs(transformedPoint.point.z - self.targetPoint.point.z) > dz:
                #     print("detected robot hand")
                #     self.targetEndEffectorPointLock.release()
                #     rospy.sleep(timeToSleep)
                #     continue
                ikreq = SolvePositionIKRequest()
                # hdr = self.targetEndEffectorPoint.header
                # hdr = Header(stamp=rospy.Time.now(), frame_id='/camera_rgb_optical_frame')
                pose = PoseStamped(
                    header=transformedPoint.header,
                    pose=Pose(
                        position=transformedPoint.point,
                        # Make the gripper point down
                        orientation=Quaternion(
                            x=orientationX,
                            y=orientationY,
                            z=orientationZ,
                            w=orientationW,
                        ),
                    ),
                )
                self.targetEndEffectorPointLock.release()
                # print("in setTargetEndEffectorPosition")
                ikreq.pose_stamp.append(pose)
                try:
                    # TODO (amal): remove the magic number 1.0
                    rospy.wait_for_service(self.iksvcStringLimb, self.timeToWait)
                    resp = self.iksvcLimb(ikreq)
                except (rospy.ServiceException, rospy.ROSException), e:
                    # rospy.logerr("IK service call failed: %s" % (e,))
                    # TODO (amal): check error handling!!!
                    print(e)
                    continue
                # resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                #                            resp.result_type)
                # TODO (amal): sometimes it says it is valid here but actually the sets are empty....
                if (resp.isValid and len(resp.joints[0].position) > 0):#(resp_seeds[0] != resp.RESULT_INVALID):
                    # seed_str = {
                    #             ikreq.SEED_USER: 'User Provided Seed',
                    #             ikreq.SEED_CURRENT: 'Current Joint Angles',
                    #             ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                    #            }.get(resp_seeds[0], 'None')
                    # print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
                    #       (seed_str,))
                    # # Format solution into Limb API-compatible dictionary
                    print("set target end effector pre acquire")
                    self.targetPositionLock.acquire()
                    print("set target end effector post acquire")
                    self.targetPosition = dict(zip(resp.joints[0].name, resp.joints[0].position))
                    self.targetPose = pose
                    self.targetPoint = transformedPoint
                    # print("target position found", self.targetPosition)
                    self.targetPositionLock.release()
                    self.shouldUpdateBaxterTargetLock.acquire()
                    self.shouldUpdateBaxterTarget = True
                    self.shouldUpdateBaxterTargetLock.release()

                else:
                    print("IK Service call failed", resp)#resp_seeds[0])
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

def main(args):
    # arg_fmt = argparse.RawDescriptionHelpFormatter
    # parser = argparse.ArgumentParser(formatter_class=arg_fmt,
    #                                  description=main.__doc__)
    # parser.add_argument(
    #     '-l', '--limb', choices=['left', 'right'], required=True,
    #     help="the limb to test"
    # )
    # parser.add_argument(
    #     '-t', '--topic', required=True,
    #     help="the limb to publish hand points to"
    # )
    # args = parser.parse_args(rospy.myargv()[1:])
    name = rospy.get_param("reachingHand/MovementController/name")
    rospy.init_node(name, anonymous=False)
    movementController = MovementController(
        limb=rospy.get_param("reachingHand/MovementController/limb"),
        topic=rospy.get_param("reachingHand/topic"),
        camera=rospy.get_param("reachingHand/camera"),
        speedRatio=rospy.get_param("reachingHand/MovementController/speedRatio"),
        jointThresholdEnd=rospy.get_param("reachingHand/MovementController/jointThreshold/end"),
        jointThresholdWarning=rospy.get_param("reachingHand/MovementController/jointThreshold/warning"),
        updateQueryRate=rospy.get_param("reachingHand/MovementController/updateQueryRate"),
        jointFilteringFactorFar=rospy.get_param("reachingHand/MovementController/jointFilteringFactor/far"),
        jointFilteringFactorClose=rospy.get_param("reachingHand/MovementController/jointFilteringFactor/close"),
        orientationX=rospy.get_param("reachingHand/MovementController/orientation/x"),
        orientationY=rospy.get_param("reachingHand/MovementController/orientation/y"),
        orientationZ=rospy.get_param("reachingHand/MovementController/orientation/z"),
        orientationW=rospy.get_param("reachingHand/MovementController/orientation/w"),
        timeToWait=rospy.get_param("reachingHand/MovementController/timeToWait"),
        name=name
    )
    # movementController.setTargetEndEffectorPosition(0.5, 0.5, 0.0)
    # rospy.sleep(1.5)
    # movementController.setTargetEndEffectorPosition(0.75, 0.5, 0.0)
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
