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

def main():
    rospy.init_node('lab_reaching_hand_kinect', anonymous=True)
    limb = 'left'
    _pub_joint_cmd = rospy.Publisher(
        '/robot/limb/' + limb + '/joint_command',
        JointCommand,
        tcp_nodelay=True,
        queue_size=1)
    _pub_speed_ratio = rospy.Publisher(
        '/robot/limb/' + limb + '/set_speed_ratio',
        Float64,
        latch=True,
        queue_size=10)
    # transform = TransformListener()
    # TODO: if multiple users, how know which hand to follow?
    _pub_speed_ratio.publish(Float64(0.5)) # Limit arm speed
    _command_msg = JointCommand()
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    pose = PoseStamped(
        header=hdr,
        pose=Pose(
            position=Point(
                x=0.5,
                y=0.5,
                z=0.0,
            ),
            orientation=Quaternion(
                x=1.0,
                y=0.0,
                z=0.0,
                w=0.0,
            ),
        ),
    )
    ikreq.pose_stamp.append(pose)
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    resp_seeds = struct.unpack('<%dB' % len(resp.result_type),
                               resp.result_type)
    if (resp_seeds[0] != resp.RESULT_INVALID):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp_seeds[0], 'None')
        print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
              (seed_str,))
        # Format solution into Limb API-compatible dictionary
        positions = dict(zip(resp.joints[0].name, resp.joints[0].position))
    # positions = {
    # "left_e0":-0.003451456772742181,
    # "left_e1":1.5270778743399294,
    # "left_s0":-0.6224127046845066,
    # "left_s1":0.03566505331833587,
    # "left_w0":,
    # "left_w1":,
    # "left_w2":,
    # }
        while not rospy.is_shutdown():
            _command_msg.names = positions.keys()
            _command_msg.command = positions.values()
            # if raw:
            _command_msg.mode = JointCommand.RAW_POSITION_MODE
            # else:
            #     _command_msg.mode = JointCommand.POSITION_MODE
            _pub_joint_cmd.publish(_command_msg)
    else:
        print(resp_seeds[0])
            # time.sleep(0.5)
        # try:
        #     (trans,rot) = transform.lookupTransform('/torso', '/left_hand_3', rospy.Time(0))
        #     print(trans, rot)
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        #     print(e)



if __name__ == '__main__':
    main()
