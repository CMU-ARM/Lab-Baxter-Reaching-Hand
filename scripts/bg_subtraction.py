#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import roslib
roslib.load_manifest('test_cam')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import baxter_interface


class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cameras/left_hand_camera/image",Image,self.callback)
    self.fgbg = cv2.BackgroundSubtractorMOG()
    self.display_pub= rospy.Publisher('/robot/xdisplay',Image)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      
    except CvBridgeError as e:
      print(e)

    img = cv_image
    fgmask = self.fgbg.apply(img)
    hands_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_hand.xml')
    
    #gray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    hands = hands_cascade.detectMultiScale(fgmask, 1.3, 5)

    for (x,y,w,h) in hands:
      cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow("Image window", fgmask)
    screen_dis =  self.bridge.cv2_to_imgmsg(fgmask, encoding="passthrough")
    self.display_pub.publish(screen_dis)
    cv2.waitKey(1)


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