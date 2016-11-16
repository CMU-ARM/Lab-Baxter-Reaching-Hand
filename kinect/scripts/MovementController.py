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
# TODO (amal): clean up these imports!!!!

class MovementController(object):
    def __init__(self, limb, topic,speedRatio, jointThresholdEnd,
        jointThresholdWarning, updateQueryRate, jointFilteringFactorFar,
        jointFilteringFactorClose, orientationX, orientationY, orientationZ,
        orientationW):
        self.limb = limb
        if limb == "left":
            self.otherLimb = "right"
        elif limb == 'right':
            self.otherLimb = "left"
        else:
            raise Exception("unkown limb, either enter left or right")
        self.pub_joint_cmd = rospy.Publisher(
            '/robot/limb/' + limb + '/joint_command',
            JointCommand,
            tcp_nodelay=True,
            queue_size=1)
        self.pub_speed_ratio = rospy.Publisher(
            '/robot/limb/' + limb + '/set_speed_ratio',
            Float64,
            latch=True,
            queue_size=1)
        self.jointAngles = dict()
        self.jointAnglesLock = threading.Lock()
        joint_state_topic = 'robot/joint_states'
        self.joint_state_sub = rospy.Subscriber(
            joint_state_topic,
            JointState,
            self.onJointStates,
            queue_size=1,
            tcp_nodelay=True)
        # transform = TransformListener()
        # TODO: if multiple users, how know which hand to follow?
        self.pub_speed_ratio.publish(Float64(speedRatio)) # Limit arm speed
        self.iksvcStringLimb = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self.iksvcLimb = rospy.ServiceProxy(self.iksvcStringLimb, SolvePositionIK)
        self.iksvcStringOtherLimb = "ExternalTools/" + self.otherLimb + "/PositionKinematicsNode/IKService"
        self.iksvcOtherLimb = rospy.ServiceProxy(self.iksvcStringOtherLimb, SolvePositionIK)
        self.shouldUpdateBaxterTarget = False
        self.shouldUpdateBaxterTargetLock = threading.Lock()
        self.targetPosition = dict()
        self.targetPoint = None
        self.killThreads = False
        self.targetPositionLock = threading.Lock()
        moveArmThread = threading.Thread(target=self.moveArm, args=(jointThresholdEnd, jointThresholdWarning, updateQueryRate, jointFilteringFactorFar, jointFilteringFactorClose))
        moveArmThread.start()
        self.endEffectorPointTopic = rospy.Subscriber(topic, PointStamped, self.endEffectorPointCallback, queue_size=1)
        self.endEffectorPoint = None
        self.endEffectorPointLock = threading.Lock()
        endEffectorThread = threading.Thread(target=self.setTargetEndEffectorPosition, args=(orientationX, orientationY, orientationZ, orientationW))
        endEffectorThread.daemon = True
        endEffectorThread.start()
        self.threads = [moveArmThread, endEffectorThread]


    def onJointStates(self, msg):
        for i, name in enumerate(msg.name):
            # print("onJointStates", i, name)
            if self.limb in name:
                self.jointAnglesLock.acquire()
                self.jointAngles[name] = msg.position[i]
                self.jointAnglesLock.release()

    def moveArm(self, jointThresholdEnd, jointThresholdWarning, updateQueryRate,
        jointFilteringFactorFar, jointFilteringFactorClose):
        rate = rospy.Rate(updateQueryRate)
        print("moveArm")
        def genf(joint, angle):
            def jointDiff():
                self.jointAnglesLock.acquire()
                if len(self.jointAngles) == 0:
                    retVal = 0
                else:
                    retVal = abs(angle - self.jointAngles[joint])
                    print(joint, retVal)
                self.jointAnglesLock.release()
                return retVal
            return jointDiff

        # Wait until we have read the joint state at least once
        # TODO (amal): do something better than a spin lock
        try:
            while not rospy.is_shutdown() and (len(self.jointAngles) == 0 or len(self.targetPosition) == 0) and not self.killThreads:
                continue
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

        def diffs():
            return [genf(j, a) for j, a in self.targetPosition.items() if j in self.jointAngles]

        reachedTarget = True
        commandMsg = JointCommand()
        try:
            while not rospy.is_shutdown() and not self.killThreads:
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
                    for diff in diffs():
                        if diff() > jointThresholdWarning:
                            allDiffsWithinWarning = False
                    if allDiffsWithinWarning and not self.closeToTarget:
                        print("WITHIN WARNING!")
                        # Don't let the target point get changed from now till the hand reaches its target
                        self.targetPositionLock.acquire()
                        print("acquired target position lock")
                        self.closeToTarget = True
                    for diff in diffs():
                        if diff() > jointThresholdEnd:
                            print("outside of threshold")
                            return True
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
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

    def endEffectorPointCallback(self, point):
        if self.endEffectorPointLock.acquire(False):
            self.endEffectorPoint = point
            self.endEffectorPointLock.release()

    def setTargetEndEffectorPosition(self, orientationX, orientationY, orientationZ, orientationW):
        listener = TransformListener()
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                self.endEffectorPointLock.acquire(True)
                # print(self.endEffectorPoint)
                if self.endEffectorPoint is None:
                    self.endEffectorPointLock.release()
                    continue
                try:
                    transformedPoint = listener.transformPoint("/base", self.endEffectorPoint)
                except (tf.ConnectivityException, tf.ExtrapolationException) as e:
                    print(e)
                    self.endEffectorPointLock.release()
                    continue
                # If the target point is at similar x and y, but the z has jumped, assume we are detecting the robot hand
                # dx = 0.05
                # dy = 0.05
                # dz = 0.1
                # if self.targetPoint is not None and abs(transformedPoint.point.x - self.targetPoint.point.x) < dx and abs(transformedPoint.point.y - self.targetPoint.point.y) < dy and abs(transformedPoint.point.z - self.targetPoint.point.z) > dz:
                #     print("detected robot hand")
                #     self.endEffectorPointLock.release()
                #     rospy.sleep(timeToSleep)
                #     continue
                ikreq = SolvePositionIKRequest()
                # hdr = self.endEffectorPoint.header
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
                self.endEffectorPointLock.release()
                # print("in setTargetEndEffectorPosition")
                ikreq.pose_stamp.append(pose)
                try:
                    # TODO (amal): remove the magic number 5.0
                    timeToWait = 1.0
                    rospy.wait_for_service(self.iksvcStringLimb, timeToWait)
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
    rospy.init_node('MovementController', anonymous=True)
    movementController = MovementController(
        limb=rospy.get_param("reachingHand/MovementController/limb"),
        topic=rospy.get_param("reachingHand/topic"),
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
