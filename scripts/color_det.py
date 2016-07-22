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
import numpy as np

class image_converter:

  def __init__(self):
   # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cameras/head_camera/image",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      
    except CvBridgeError as e:
      print(e)

    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    boundaries = [
    #([17, 15, 100], [14, 21, 246])#,
    ([0,86,28],[0,42,80])
    #([86, 31, 4], [220, 88, 50]),
    #([25, 146, 190], [62, 174, 250]),
    #([103, 86, 65], [145, 133, 128])
    ]
    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
     
        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(cv_image, cv_image, mask = mask)
     
        
    cv2.imshow("images", np.hstack([cv_image, output]))
    #cv2.imshow("images", mask)
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