#! /usr/bin/env python

import rospy

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the fibonacci action, including the
# goal message and the result message.
import rospy
from lab_reaching_hand_kinect.msg import (
    ReachingHandAction,
    ReachingHandGoal,
)


def movementControllerClient():
    # Creates the SimpleActionClient, passing the type of the action
    # (FibonacciAction) to the constructor.
    # while True:
    name=rospy.get_param("reachingHand/MovementController/name")
    client = actionlib.SimpleActionClient(name, ReachingHandAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server(timeout=rospy.Duration(secs=5.0))

    # Creates a goal to send to the action server.
    goal = ReachingHandGoal()

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A FibonacciResult

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('fibonacci_client_py')
        result = movementControllerClient()
        print("Result:", result)
    except rospy.ROSInterruptException:
        print "program interrupted before completion"
