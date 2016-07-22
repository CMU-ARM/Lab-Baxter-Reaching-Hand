#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('test_cam')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from matplotlib import pyplot as plt

class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub_1 = message_filters.Subscriber("/cameras/left_hand_camera/image",Image)
    self.image_sub_2 = message_filters.Subscriber("/cameras/right_hand_camera/image",Image)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub_1, self.image_sub_2], 1,1)
    self.ts.registerCallback(self.callback)

  def callback(self,image_1, image_2):
    try:
      cv_image_1 = self.bridge.imgmsg_to_cv2(image_1, "bgr8")
      cv_image_2 = self.bridge.imgmsg_to_cv2(image_2, "bgr8")

    except CvBridgeError as e:
      print(e)

    cv2.imshow('left_hand_camera 1',cv_image_1)
    cv2.imshow('right_hand_camera 2',cv_image_2)
    cv2.waitKey(1)

    stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
    gray_1 = cv2.cvtColor(cv_image_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(cv_image_2, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(gray_1,gray_2)

    plt.imshow(disparity,'gray')
    plt.show()

    #cv2.waitKey(1)


def main(args):

  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)