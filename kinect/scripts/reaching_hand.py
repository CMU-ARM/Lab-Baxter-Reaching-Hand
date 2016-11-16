#!/usr/bin/env python
import rospy
import tf
from tf import TransformListener
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
)
from std_msgs.msg import Header
import struct
import time
from MovementController import MovementController
from HandDetector import HandDetector

def main():
    # We don't want multiple instances of this node running
    rospy.init_node('lab_reaching_hand_kinect', anonymous=False)
    limb = "left"
    rate = rospy.Rate(2)
    dTime = 2.5
    listener = TransformListener()
    movementController = MovementController(limb)
    handDetector = HandDetector()
    try:
        while not rospy.is_shutdown():
            # TODO (amal): maybe change it so instead of getting xyz here we get a poseStamped?
            # coord = handDetector.getAveragePosByTime(dTime)
            # print("getAveragePosByTime returns ", coord)
            # if coord is not None and coord[0] is not None:
                # print("in if statement")
                # pointMsg = PointStamped()
                # # TODO (amal): allow this camera name to be flexible
                # pointMsg.header = Header(stamp=rospy.Time.now(), frame_id='/camera_link')
                # pointMsg.point = Point()
                # pointMsg.point.x = coord[0].x
                # pointMsg.point.y = coord[0].y
                # pointMsg.point.z = coord[0].z
                # TODO (amal): except tf.ConnectivityException
                # try:
                #     transformedPoint = listener.transformPoint("/base", coord[0])
                # except (tf.ConnectivityException, tf.ExtrapolationException) as e:
                #     print(e)
                #     continue
                # print("transformed point", transformedPoint)
                # movementController.setTargetEndEffectorPosition(transformedPoint.point.x, transformedPoint.point.y, transformedPoint.point.z)
                rate.sleep()
    except KeyboardInterrupt, rospy.ROSInterruptException:
        pass
    print("quitting")
    movementController.killThreads = True
    handDetector.killThreads = True
    # movementController.setTargetEndEffectorPosition(0.5, 0.5, 0.0)
    # time.sleep(0.5)
    # movementController.setTargetEndEffectorPosition(0.75, 0.5, 0.0)
    # try:
    #   rospy.spin()
    # except KeyboardInterrupt, rospy.ROSInterruptException:
    #   print("Shutting down")

if __name__ == '__main__':
    main()
