#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Example
"""
import argparse
import struct
import sys

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
import baxter_interface
import moveit_commander
import moveit_msgs.msg

import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)


def ik_test(limb):
    rospy.init_node("rsdk_ik_service_client")
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        'left': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=0.657579481614,
                    y=0.851981417433,
                    z=0.0388352386502,
                ),
                orientation=Quaternion(
                    x=-0.366894936773,
                    y=0.885980397775,
                    z=0.108155782462,
                    w=0.262162481772,
                ),
            ),
        ),
        'right': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=0.656982770038,
                    y=-0.852598021641,
                    z=0.0388609422173,
                ),
                orientation=Quaternion(
                    x=0.367048116303,
                    y=0.885911751787,
                    z=-0.108908281936,
                    w=0.261868353356,
                ),
            ),
        ),
    }

    ikreq.pose_stamp.append(poses[limb])
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1

    # Check if result valid, and type of seed ultimately used to get solution
    # convert rospy's string representation of uint8[]'s to int's
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
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        print "\nIK Joint Solution:\n", limb_joints
        print "------------------"
        print "Response Message:\n", resp

        jtaClient = actionlib.SimpleActionClient(
            'robot/limb/' + limb + "/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

        group = moveit_commander.MoveGroupCommander(limb+"_arm")
        group.set_planner_id("LBKPIECE1")
        group.clear_path_constraints()
        group.clear_pose_targets()
        group.set_pose_target(poses[limb].pose)
        trajectory = group.plan()
        print(trajectory, len(trajectory.joint_trajectory.points))

        server_up = jtaClient.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running code.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        downsampleFactor = 10
        interuptAfter = 1 # for loop iteration
        for i in xrange(len(trajectory.joint_trajectory.points)/downsampleFactor+1):
            if i == interuptAfter:
                interupt(limb, group, jtaClient)
                break
            goal = FollowJointTrajectoryGoal()
            goal.goal_time_tolerance = rospy.Time(0.5)
            goal.trajectory.joint_names = [limb + '_' + joint for joint in \
                ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
            goal.trajectory.points = trajectory.joint_trajectory.points[i*downsampleFactor:min((i+1)*downsampleFactor, len(trajectory.joint_trajectory.points))]
            goal.trajectory.header.stamp = rospy.Time.now()
            jtaClient.send_goal(goal)
            jtaClient.wait_for_result(timeout=rospy.Duration(15.0))
            result = jtaClient.get_result()
            print("i: ", i, "Got result: ", result)

        # arm = baxter_interface.Limb(limb)
        # while not rospy.is_shutdown():
        #     arm.set_joint_positions(limb_joints)
        #     rospy.sleep(0.01)

        # group.set_joint_value_target(limb_joints)

        # TODO: check if len of points = 0
        # print(trajectory, len(trajectory.joint_trajectory.points), trajectory.joint_trajectory.points[0], trajectory.joint_trajectory.points[1])
        # i = 0#len(trajectory.joint_trajectory.points)-1
        # status = group.go(joints=trajectory.joint_trajectory.points[i], wait=True)
        # print("moved to first position")
        # rospy.sleep(5.)
        # status = group.go(joints=trajectory.joint_trajectory.points[-1], wait=True)
        # print("moved to last position")
        # while i > 0:
        #     i+=1
        #     status = group.go(joints=trajectory.joint_trajectory.points[i], wait=True)
        # status = group.go(wait=True)
        # print(status)
        # while not status:trajectory.joint_trajectory.points[i]
        #     status = group.go(wait=True)
        #     print(status)
        # print("left while loop")
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")

    return 0

def interupt(limb, group, jtaClient):
    pose = Pose(
        position=Point(
            x=0.5,
            y=0.5,
            z=0.5,
        ),
        orientation=Quaternion(
            x=-0.1,
            y=0.1,
            z=0.1,
            w=0.1,
        ),
    )

    group.clear_path_constraints()
    group.clear_pose_targets()
    group.set_pose_target(pose)
    trajectory = group.plan()

    goal = FollowJointTrajectoryGoal()
    goal.goal_time_tolerance = rospy.Time(0.5)
    goal.trajectory.joint_names = [limb + '_' + joint for joint in \
        ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]
    goal.trajectory.points = trajectory.joint_trajectory.points
    goal.trajectory.header.stamp = rospy.Time.now()
    jtaClient.send_goal(goal)
    jtaClient.wait_for_result(timeout=rospy.Duration(15.0))
    result = jtaClient.get_result()
    print("interrupt.  Got result:", result)


def main():
    """RSDK Inverse Kinematics Example

    A simple example of using the Rethink Inverse Kinematics
    Service which returns the joint angles and validity for
    a requested Cartesian Pose.

    Run this example, passing the *limb* to test, and the
    example will call the Service with a sample Cartesian
    Pose, pre-defined in the example code, printing the
    response of whether a valid joint solution was found,
    and if so, the corresponding joint angles.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        '-l', '--limb', choices=['left', 'right'], required=True,
        help="the limb to test"
    )
    print("after add arg")
    args = parser.parse_args(rospy.myargv()[1:])
    print("parsed args")

    return ik_test(args.limb)

if __name__ == '__main__':
    main()
