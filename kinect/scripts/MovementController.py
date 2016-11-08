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
    def __init__(self, limb, topic,speedRatio=0.3):
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
        joint_state_topic = 'robot/joint_states'
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
        self.iksvc = rospy.ServiceProxy(self.iksvcString, SolvePositionIK)
        self.shouldUpdateBaxterTarget = False
        self.shouldUpdateBaxterTargetLock = threading.Lock()
        self.targetPosition = dict()
        self.killThreads = False
        self.targetPositionLock = threading.Lock()
        moveArmThread = threading.Thread(target=self.moveArm, args=())
        moveArmThread.start()
        self.endEffectorPointTopic = rospy.Subscriber(topic, PointStamped, self.endEffectorPointCallback, queue_size=1)
        self.endEffectorPoint = None
        self.endEffectorPointLock = threading.Lock()
        endEffectorThread = threading.Thread(target=self.setTargetEndEffectorPosition, args=())
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

    def moveArm(self, threshold=settings.JOINT_ANGLE_TOLERANCE, timeToSleep=2):
        print("moveArm")
        def genf(joint, angle):
            def jointDiff():
                self.jointAnglesLock.acquire()
                if len(self.jointAngles) == 0:
                    retVal = 0
                else:
                    retVal = abs(angle - self.jointAngles[joint])
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
            while not rospy.is_shutdown() and not self.killThreads: # TODO (amal): have a rospy.is_shutdown check in this loop!
                if reachedTarget:
                    self.shouldUpdateBaxterTargetLock.acquire()
                    if not self.shouldUpdateBaxterTarget:
                        self.shouldUpdateBaxterTargetLock.release()
                        rospy.sleep(timeToSleep)
                        continue
                    self.shouldUpdateBaxterTargetLock.release()
                # self.targetPositionLock.acquire()
                # positions = self.targetPosition
                # self.targetPositionLock.release()

                # TODO (amal): look into exponential moving average
                def filtered_cmd():
                    print(self.targetPosition, self.jointAngles)
                    self.targetPositionLock.acquire()
                    retPositions = dict()
                    # First Order Filter - ????? Hz Cutoff
                    # factor = 0.012488
                    factor = 0.1
                    for joint in  self.targetPosition.keys():
                        retPositions[joint] = factor * self.targetPosition[joint] + (1-factor) * self.jointAngles[joint] # Get most up to date value
                    self.targetPositionLock.release()
                    return retPositions

                def loopGuard():
                    if rospy.is_shutdown() or self.killThreads:
                        return False
                    for diff in diffs():
                        if diff() > threshold:
                            return True
                    return False
                try:
                    while loopGuard() :
                        position = filtered_cmd()
                        commandMsg.names = position.keys()
                        commandMsg.command = position.values()
                        # commandMsg.command = self.targetPosition.values()
                        print(self.jointAngles, self.targetPosition, position)
                        commandMsg.mode = JointCommand.RAW_POSITION_MODE
                        self.pub_joint_cmd.publish(commandMsg)
                except KeyboardInterrupt, rospy.ROSInterruptException:
                    return
                reachedTarget = True
                for diff in diffs():
                    if diff() > threshold:
                        continue
                self.shouldUpdateBaxterTargetLock.acquire()
                self.shouldUpdateBaxterTarget = False
                self.shouldUpdateBaxterTargetLock.release()
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

    def endEffectorPointCallback(self, point):
        if self.endEffectorPointLock.acquire(False):
            self.endEffectorPoint = point
            self.endEffectorPointLock.release()

    def setTargetEndEffectorPosition(self):
        listener = TransformListener()
        try:
            while not rospy.is_shutdown() and not self.killThreads:
                self.endEffectorPointLock.acquire(True)
                print(self.endEffectorPoint)
                if self.endEffectorPoint is None:
                    self.endEffectorPointLock.release()
                    continue
                try:
                    transformedPoint = listener.transformPoint("/base", self.endEffectorPoint)
                except (tf.ConnectivityException, tf.ExtrapolationException) as e:
                    print(e)
                    self.endEffectorPointLock.release()
                    continue
                ikreq = SolvePositionIKRequest()
                # hdr = self.endEffectorPoint.header
                # hdr = Header(stamp=rospy.Time.now(), frame_id='/camera_rgb_optical_frame')
                pose = PoseStamped(
                    header=transformedPoint.header,
                    pose=Pose(
                        position=transformedPoint.point,
                        # Make the gripper point down
                        orientation=Quaternion(
                            x=1.0,
                            y=0.0,
                            z=0.0,
                            w=0.0,
                        ),
                    ),
                )
                self.endEffectorPointLock.release()
                # print("in setTargetEndEffectorPosition")
                ikreq.pose_stamp.append(pose)
                try:
                    # TODO (amal): remove the magic number 5.0
                    rospy.wait_for_service(self.iksvcString, 1.0)
                    resp = self.iksvc(ikreq)
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
                    self.targetPositionLock.acquire()
                    self.targetPosition = dict(zip(resp.joints[0].name, resp.joints[0].position))
                    print("target position found", self.targetPosition)
                    self.targetPositionLock.release()
                    self.shouldUpdateBaxterTargetLock.acquire()
                    self.shouldUpdateBaxterTarget = True
                    self.shouldUpdateBaxterTargetLock.release()

                else:
                    print("IK Service call failed", resp)#resp_seeds[0])
        except KeyboardInterrupt, rospy.ROSInterruptException:
            return

def main(args):
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-l', '--limb', choices=['left', 'right'], required=True,
        help="the limb to test"
    )
    parser.add_argument(
        '-t', '--topic', required=True,
        help="the limb to publish hand points to"
    )
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('MovementController', anonymous=True)
    movementController = MovementController(args.limb, args.topic)
    # movementController.setTargetEndEffectorPosition(0.5, 0.5, 0.0)
    # rospy.sleep(1.5)
    # movementController.setTargetEndEffectorPosition(0.75, 0.5, 0.0)
    try:
        rospy.spin()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
