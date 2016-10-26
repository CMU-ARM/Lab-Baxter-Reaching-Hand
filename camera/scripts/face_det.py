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
from lab_baxter_common.camera_control_helpers import CameraController
import os.path


class image_converter:

  def __init__(self):
    # Safely turn the head camera on
    CameraController.openCameras("head_camera")

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cameras/head_camera/image",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    except CvBridgeError as e:
      print(e)

    face_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_eye.xml')

    img = cv_image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
	print("found face at", x, y, w, h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("Image window", img)
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
