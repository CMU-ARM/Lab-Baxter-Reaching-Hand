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
import time 


class image_converter:

  def __init__(self):
    self.img_2 = None
    self.bridge = CvBridge()
    #self.image_sub_right = rospy.Subscriber("/cameras/right_hand_camera/image",Image,self.callback_right)
    self.image_sub = rospy.Subscriber("/cameras/left_hand_camera/image",Image,self.callback)
    #self.display_pub= rospy.Publisher('/robot/xdisplay',Image)
    self.fgbg = cv2.BackgroundSubtractorMOG()
    self.x = 0
    self.y = 0
    self.z = 0
    #self.display_pub= rospy.Publisher('/robot/xdisplay',Image)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      
    except CvBridgeError as e:
      print(e)
       
    img = cv_image
    converted = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB) # Convert image color scheme to YCrCb
    min_YCrCb = np.array([0,133,77],np.uint8) # Create a lower bound for the skin color
    max_YCrCb = np.array([255,173,127],np.uint8) # Create an upper bound for skin color

    skinRegion = cv2.inRange(converted,min_YCrCb,max_YCrCb) # Create a mask with boundaries
    skinMask = cv2.inRange(converted,min_YCrCb,max_YCrCb) # Duplicate of the mask for comparison
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # Apply a series of errosion and dilations to the mask
    skinMask = cv2.erode(skinMask, kernel, iterations = 2) # Using an elliptical Kernel
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) # Blur the image to remove noise
    skin = cv2.bitwise_and(img, img, mask = skinMask) # Apply the mask to the frame

    height, width, depth = cv_image.shape  
    self.frame = cv_image.shape 
    hands_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_hand.xml')
    hands = hands_cascade.detectMultiScale(img, 1.3, 5) # Detect the hands on the converted image
    
    for (x,y,z,h) in hands: # Get the coordinates of the hands
      d = h/2
      self.x = x+d
      self.y = y+d
      # If the hands coordinates are within the range of the skin color
      if (min_YCrCb[0] < converted[self.y,self.x,0] < max_YCrCb[0] and min_YCrCb[1] < converted[self.y,self.x,1] < max_YCrCb[1] 
        and min_YCrCb[2] < converted[self.y,self.x,2] < max_YCrCb[2]):
        #if 0.10*width < x < 0.9*width and 0.10*height < y < 0.9*height: # Removes noise 
        cv2.circle(img,(self.x,self.y),50,(0,0,255),5) # Circle the detected hand

    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contour on the skin detection
    for i, c in enumerate(contours): # Draw the contour on the source frame
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(img, contours, i, (255, 255, 0), 2)
    cv2.circle(img,(640,200),50,(255,0,255),5) 

    cv2.imshow("Hand Detection", img)#np.hstack([img, self.img_2]))
    cv2.waitKey(1)

  def callback_right(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      
    except CvBridgeError as e:
      print(e)
       
    img = cv_image
    converted = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB) # Convert image color scheme to YCrCb
    min_YCrCb = np.array([0,133,77],np.uint8) # Create a lower bound for the skin color
    max_YCrCb = np.array([255,173,127],np.uint8) # Create an upper bound for skin color

    skinRegion = cv2.inRange(converted,min_YCrCb,max_YCrCb) # Create a mask with boundaries
    skinMask = cv2.inRange(converted,min_YCrCb,max_YCrCb) # Duplicate of the mask for comparison
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # Apply a series of errosion and dilations to the mask
    skinMask = cv2.erode(skinMask, kernel, iterations = 2) # Using an elliptical Kernel
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) # Blur the image to remove noise
    skin = cv2.bitwise_and(img, img, mask = skinMask) # Apply the mask to the frame

    height, width, depth = cv_image.shape  
    self.frame = cv_image.shape 
    hands_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_hand.xml')
    hands = hands_cascade.detectMultiScale(img, 1.3, 5) # Detect the hands on the converted image
    
    for (x,y,z,h) in hands: # Get the coordinates of the hands
      d = h/2
      self.x = x+d
      self.y = y+d
      # If the hands coordinates are within the range of the skin color
      if (min_YCrCb[0] < converted[self.y,self.x,0] < max_YCrCb[0] and min_YCrCb[1] < converted[self.y,self.x,1] < max_YCrCb[1] 
        and min_YCrCb[2] < converted[self.y,self.x,2] < max_YCrCb[2]):
        #if 0.10*width < x < 0.9*width and 0.10*height < y < 0.9*height: # Removes noise 
        cv2.circle(img,(self.x,self.y),50,(0,0,255),5) # Circle the detected hand

    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contour on the skin detection
    for i, c in enumerate(contours): # Draw the contour on the source frame
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(img, contours, i, (255, 255, 0), 2)
    self.img_2 = img


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