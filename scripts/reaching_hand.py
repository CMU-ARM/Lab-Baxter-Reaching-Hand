#!/usr/bin/env python
import rospy
import sys
import cv2
import time
import tf
from tf import TransformListener
import numpy as np
import math
from numpy.linalg import inv
import roslib
from baxter_pykdl import baxter_kinematics
import baxter_interface
from baxter_interface import Limb
import image_geometry
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from baxter_interface import CHECK_VERSION
from sensor_msgs.msg import Range
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
import struct
from operator import mul
from itertools import imap

class get_pos(object):

  def __init__(self,limb_viewer,limb_checker,parameter_reset):
    '''
    Viewer - Limb that will end up reaching out to the user's hand
    Checker - Limb that checks whether the user's hand is still in position
    Parameter Reset - Determines whether the camera parameters should be reset. Reset is only needed once
    '''
    #Initializing for the translation
    self.left_arm = Limb('left')
    self.right_arm = Limb('right')
    self.limb = limb_viewer 
    self.img_viewer = None
    self.arm_viewer = baxter_interface.Limb(limb_viewer) # Assinging the viewer to the correct limb
    self.arm_checker = baxter_interface.Limb(limb_checker) # Assigning the checker to the correct limb
    self.hand_area = 0 # Hand area is used by Checker to determine whether the hand is still detected
    self.bridge = CvBridge() # ROS to Opencv
    # Coordinates and global variables setup
    self.torque = None # Torque will be used as a detection method
    self.frame = None # Will be used for image shape
    self.x = 0 # Will be assigned to x-pixel-coordinate of detected hand // Will later be converted to base frame
    self.y = 0 # Will be assigned to y-pixel-coordinate of detected hand // Will later be converted to base frame
    self.z_camera = 0 # Will be assigned to z spatial coordinate WRT to camera
    # Arm sensor setup
    self.distance = {}
    root_name = "/robot/range/"
    sensor_name = ["left_hand_range/state","right_hand_range/state"]
    # Assigning the camera topics to viewer and checker depending on the user input
    if limb_viewer == 'left':
        self.camera_viewer = 'left_hand_camera'
        camera_checker = 'right_hand_camera'
        # Subscribing to the left sensor
        self.__sensor  = rospy.Subscriber(root_name + sensor_name[0],Range, callback=self.sensorCallback, callback_args="left",queue_size=1)
    else: 
        self.camera_viewer = 'right_hand_camera'
        camera_checker = 'left_hand_camera'
        # Subscribing to the right sensor
        self.__sensor  = rospy.Subscriber(root_name + sensor_name[1],Range, callback=self.sensorCallback, callback_args="right",queue_size=1)
    # resetting the parameters of the viewer and checker if instructed
    if parameter_reset == True:
        self.left_camera = baxter_interface.CameraController(self.camera_viewer)
        self.left_camera.resolution = (640,400)
        self.left_camera.exposure = -1
        print "Viewer-camera parameter check"
        self.right_camera = baxter_interface.CameraController(camera_checker)
        self.right_camera.resolution = (640,400)
        self.right_camera.exposure = -1
        print "Checker-camera parameter check"
    # Subscribing to the cameras
    self.viewer_image_sub = rospy.Subscriber("/cameras/" + self.camera_viewer + "/image",Image,self.callback_viewer) # Subscribing to Camera
    self.checker_image_sub = rospy.Subscriber("/cameras/" + camera_checker + "/image",Image,self.callback_checker) # Subscribing to Camera
    # Rotation matrix set up
    self.tf = TransformListener()
    # Orientation of the shaker will determine how the viewer's orientation will be once it reaches its final position
    self.left_orientation_shaker = [-0.520, 0.506, -0.453, 0.518] # Defined orientation // Obtained through tf_echo // ****GRIPPER****
    self.right_orientation_shaker = [0.510, 0.550, 0.457, 0.478]
    self.camera_gripper_offset = [0.038, 0.012, -0.142] # Offset of camera from the GRIPPER reference frame
    self._cur_joint = {
     'left_w0': 0.196733035822, 
     'left_w1': 0.699878733673, 
     'left_w2': 0,
     'left_e0': -0.303344700458, 
     'left_e1': 1.90098568922, 
     'left_s0': -0.263844695215, 
     'left_s1': -1.03467004025}

  def follow_up(self,joint_solution=None):
    '''
    Any follow up instructions, after the hand is reached, should be in here
    '''
    if joint_solution == None:
        joint_solution = self._cur_joint
    self.__sensor.unregister() #
    self.viewer_image_sub.unregister() 
    self.checker_image_sub.unregister()
    self.arm_viewer.move_to_joint_positions(joint_solution)

  def sensorCallback(self,msg,side):
    self.distance[side] = msg.range
    if msg.range < 0.05: # Only assigns a value to sensor if the hand is less than 10cm away
        self.z_camera = msg.range # Assign the z-coordinate
    else:
        self.z_camera = None

  def callback_viewer(self,data):
    '''
    This is the callback function of the viewer, i.e, the thread that the viewer creates once it's initialized.
    The viewer callback first runs a skin color detection, creates a mask from the given color range, and then
    the hand detection is ran on the mask. The hand detection is done through a cascade classifier, and the 
    coordinates of the hands are assigned to the global variables x and y. To be noted that the contour drawing
    is only to provide a good display; it doesn't affect the skin detection, though it uses the same mask
    '''
    try:
      # Creates an OpenCV image from the ROS image
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # The torque is used as a method of checking whether the arm has reached the user
    self.torque = self.arm_viewer.endpoint_effort()  
    self.torque = self.torque_mag(self.torque) # The torque assigned is the magnitude of the torque detected

    img = cv_image
    converted = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB) # Convert image color scheme to YCrCb
    min_YCrCb = np.array([0,133,77],np.uint8) # Create a lower bound for the skin color
    max_YCrCb = np.array([255,173,127],np.uint8) # Create an upper bound for skin color

    skinRegion = cv2.inRange(converted,min_YCrCb,max_YCrCb) # Create a mask with boundaries
    
    skinMask = cv2.inRange(converted,min_YCrCb,max_YCrCb) # Duplicate of the mask f
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # Apply a series of errosion and dilations to the mask
    #skinMask = cv2.erode(skinMask, kernel, iterations = 2) # Using an elliptical Kernel
    #skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) # Blur the image to remove noise
    skin = cv2.bitwise_and(img, img, mask = skinMask) # Apply the mask to the frame

    height, width, depth = cv_image.shape # Obtain the dimensions of the image
    self.frame = cv_image.shape
    hands_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_hand.xml')
    hands = hands_cascade.detectMultiScale(skinMask, 1.3, 5) # Detect the hands on the converted image
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contour on the skin detection
    for i, c in enumerate(contours): # Draw the contour on the source frame
        area = cv2.contourArea(c)
        if area > 10000: # Noise isn't circled
            cv2.drawContours(img, contours, i, (255, 255, 0), 2)
    for (x,y,z,h) in hands: # Get the coordinates of the hands
      d = h/2
      self.x = x+d # Gets the center of the detected hand
      self.y = y+d # Gets the center of the detected hand
      cv2.circle(img,(self.x,self.y),50,(0,0,255),5) # Circle the detected hand
    self.img_viewer = img

  def callback_checker(self,data):
    '''
    This is the callback function of the checker, i.e, the thread that the checker creates once it's initialized. 
    The checker callback runs in the same way as the viewer callback, but its main use is to ensure that a hand
    is still detected. It does so by checking the area of the contour drawn on the image, and hence, unlike the
    viewer callback, the contour affects the hand detection. The contour area is however unstable, and might not
    produce the best results. The skin color detection has a high range of red color as wll, making the contour 
    detection less stable when a red colopr is in range. Hence that is why the exposure of the checker is decreased
    so as to reduce the noise colors.
    '''
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
    #skinMask = cv2.erode(skinMask, kernezl, iterations = 2) # Using an elliptical Kernel
    #skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) # Blur the image to remove noise
    skin = cv2.bitwise_and(img, img, mask = skinMask) # Apply the mask to the frame

    height, width, depth = cv_image.shape  
    hands_cascade = cv2.CascadeClassifier('/home/steven/ros_ws/src/test_cam/haarcascade_hand.xml')
    hands = hands_cascade.detectMultiScale(skinMask, 1.3, 5) # Detect the hands on the converted image
    for (x,y,z,h) in hands: # Get the coordinates of the hands
      d = h/2
      x = x+d
      y = y+d
      cv2.circle(img,(x,y),50,(0,0,255),5)

    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the contour on the skin detection
    self.hand_area = 0
    for i, c in enumerate(contours): # Draw the contour on the source frame
        area = cv2.contourArea(c)
        if area > 10000:
            cv2.drawContours(img, contours, i, (255, 255, 0), 2) 
            self.hand_area = area# - area_1
    
    # The following function will create a contour based on the red color scheme. This function should be enabled whenever 
    # a red color is detected by the checker. The red color detected will alter the hand area detected, and hence will
    # detect an large area even if a hand is not in range, area corresponding to the red color. Hence the red area should be subtracted
    # from the entire detected area, to obtain the actual area of the hand
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 10, 10])
    upper_red = np.array([10, 240, 240])
    
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red = cv2.bitwise_and(img, img, mask = red_mask)

    cv2.imshow("Viewer Camera // Checker Camera",np.hstack([img, self.img_viewer]))
    cv2.waitKey(1)

  def torque_mag(self,a):
    # Gets the magnitude of the torque detected
    mag_a = math.sqrt((a['force'].z*a['force'].z)+(a['force'].y*a['force'].y)+(a['force'].x*a['force'].x))
    return mag_a

  def get_starting_pos(self):
    # Sets the limbs to their correct starting position
    if self.limb == 'left':
        self.left_arm.move_to_joint_positions(dict({
            'left_e0':-1.1339952974442922,
            'left_e1':1.2954467753692318,
            'left_s0':0.8252816638823526, 
            'left_s1':0.048703890015361885,
            'left_w0':-0.14879613642488512,
            'left_w1':1.176179769111141,
            'left_w2':0.5867476513661708}), timeout = 15.0)  
        self.right_arm.move_to_joint_positions(dict({
            'right_e0':1.0438739261560241,
            'right_e1':1.2797234722934063,
            'right_s0':-0.1257864246066039,
            'right_s1':-0.3171505278953093, 
            'right_w0':0.3186845086831947, 
            'right_w1':1.1278593742927505,
            'right_w2':-0.215907795894872}), timeout = 15.0) 
    else:
        self.left_arm.move_to_joint_positions(dict({
            'left_e0':-1.2475098757478127,
            'left_e1':1.1830826826566254,
            'left_s0':0.292990330486114,
            'left_s1':-0.12540292940963257,
            'left_w0':-0.024927187803137973,
            'left_w1':1.301966193717745, 
            'left_w2':0.15953400194008302}), timeout = 15.0)  
        self.right_arm.move_to_joint_positions(dict({
            'right_e0':1.0933448065653286,
            'right_e1':1.2609322076418101,
            'right_s0':-0.6024709544419963, 
            'right_s1':-0.08053399136398422,
            'right_w0':0.2906893593042859,
            'right_w1':1.212995308020391,
            'right_w2':-0.19251458887961942}), timeout = 15.0) 

  def unit_vector_to_point(self,x,y):
    '''
    Creates a unit vector from the camera's frame, given a pixel point from the image
    '''
    height, width, depth = self.frame # Obtain the dimensions of the frame
    cam_info = rospy.wait_for_message("/cameras/"+self.camera_viewer+"/camera_info", CameraInfo, timeout=None)
    img_proc = PinholeCameraModel()
    img_proc.fromCameraInfo(cam_info)
    # The origin of the camera is initally set to the top left corner
    # However the projectPixelTo3dRay uses the origin at the center of the camera. Hence the coordinates have to be converted
    x = x - width/2
    y = height/2 - y
    # Creates a unit vector to the given point. Unit vector passes through point from the center of the camera
    n = img_proc.projectPixelTo3dRay((x,y))
    return n 

  def unit_vector_gripper_frame(self,u_vector):
    '''
    Converts the unit vector from the camera's frame to that of the gripper
    '''
    pose = self.arm_viewer.endpoint_pose()
    pos_gripper = [pose['position'].x,pose['position'].y,pose['position'].z]
    u_vector_gripper = [u_vector[0]+self.camera_gripper_offset[0],u_vector[1]+self.camera_gripper_offset[1],u_vector[2]+self.camera_gripper_offset[2]]
    return u_vector_gripper

  def align(self,alignment_vector=None):
    '''
    To ensure that a inverse kinematics will return a valid joint solution set, a first alignment is needed.
    Aligning the gripper with the unit vector to the hand's position will ensure that. 
    '''
    if alignment_vector != None:
        alignment_vector = alignment_vector[:3] # Ensure that vector is a 3x1
    height, width, depth = self.frame # Gets the dimension of the image
    pose = self.arm_viewer.endpoint_pose() # Gets the current pose of the end effector 
    pos = [pose['position'].x,pose['position'].y,pose['position'].z] # Gets the position
    quat = [pose['orientation'].x,pose['orientation'].y,pose['orientation'].z,pose['orientation'].w] # Gets the orientation
    __matrix = self.tf.fromTranslationRotation(pos,quat) # Rotational matrix is obtained from pos and quat
    __matrix = __matrix[:3,:3] # __matrix contain the rotational, and translational component, alongside with a last row of 0,0,0,1 i.e matrix is 4x4
    # Only the rotational part is needed for alignement purposes, i.e a 3x3
    
    def alignment_to_vec(b,a): # b = z-axis vector // a = alignment vector (unit vector to point)
        '''
        Returns the rotational matrix, that wil align the vector b to a
        '''
        a_dot_b = sum(imap(mul, a, b))#np.dot(a,b)
        n_a = np.array([float(a[0]),float(a[1]),float(a[2])])
        n_b = np.array([float(b[0]),float(b[1]),float(b[2])])
        a_x_b = np.cross(n_a,n_b)
        a_x_b = np.matrix([[float(a_x_b[0])],[float(a_x_b[1])],[float(a_x_b[2])]])
        mod_a = math.sqrt((float(a[0])*float(a[0]))+(float(a[1])*float(a[1]))+(float(a[2])*float(a[2])))
        mod_b = math.sqrt((float(b[0])*float(b[0]))+(float(b[1])*float(b[1]))+(float(b[2])*float(b[2])))   
        mod_a_x_b = math.sqrt((float(a_x_b[0])*float(a_x_b[0]))+(float(a_x_b[1])*float(a_x_b[1]))
                        +(float(a_x_b[2])*float(a_x_b[2])))
        x = a_x_b/mod_a_x_b
        alpha = a_dot_b/(mod_a*mod_b)
        theta = math.acos(alpha)
        a_matrix = np.matrix([[0,float(-x[2]),float(x[1])],[float(x[2]),0,float(-x[0])],[float(-x[1]),float(x[0]),0]])
        rotation_matrix = np.identity(3)+(math.sin(theta)*a_matrix)+((1-(math.cos(theta)))*(np.dot(a_matrix,a_matrix)))
        return rotation_matrix

    def create_from_rot_matrix(rot):
        """
        Rotation matrix created from quaternion
        Create from rotation matrix,
        modified from
        https://github.com/CMU-ARM/ARBT-Baxter-Nav-Task/blob/stable/scripts/Quaternion.py
        """
        tr = np.trace(rot)
        if (tr > 0):
            s = np.sqrt(tr + 1) * 2
            w = 0.25 * s
            x = (rot[2,1] - rot[1,2])/s
            y = (rot[0,2] - rot[2,0])/s
            z = (rot[1,0] - rot[0,1])/s
        elif(rot[0,0] > rot[1,1] and rot[0,0] > rot[2,2]):
            s = np.sqrt(1 + rot[0,0] - rot[1,1] - rot[2,2]) * 2
            w = (rot[2,1] - rot[1,2])/s
            x = 0.25 * s
            y = (rot[0,1] + rot[1,0])/s
            z = (rot[0,2] + rot[2,0])/s
        elif(rot[1,1] > rot[2,2]):
            s = np.sqrt(1 + rot[1,1] - rot[0,0] - rot[2,2]) * 2
            w = (rot[0,2] - rot[2,0])/s
            x = (rot[0,1] + rot[1,0])/s
            y = 0.25 * s
            z = (rot[1,2] + rot[2,1])/s
        else:
            s = np.sqrt(1 + rot[2,2] - rot[1,1] - rot[0,0]) * 2
            w = (rot[1,0] - rot[0,1])/s
            x = (rot[0,2] + rot[2,0])/s
            y = (rot[1,2] + rot[2,1])/s
            z = 0.25 * s
        return x,y,z,w

    def hamilton_product(b,a):
        q = [x,y,z,w]
        q[3] = a[3] * b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]
        q[0] = a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1]
        q[1] = a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0]
        q[2] = a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3]
        return q

    z_vector = np.dot(__matrix,np.matrix([[0],[0],[1]])) # Converts the z-axis of the camera to the base frame
    print "Z-vector BF: ",z_vector
    rotation_matrix = alignment_to_vec(z_vector,alignment_vector) # Rotation matrix that aligns the z-axis to the unit vector, pointing towards the hand
    x,y,z,w = create_from_rot_matrix(rotation_matrix) # Gets the orientation of alignement
    rot_quat = [x,y,z,w]
    final_quat = hamilton_product(quat,rot_quat) # Gets the final orientation of alignement
    self.ik(self.limb,pos,final_quat,True) # Aligns the viewer
    print "Aligned"

  def pixel_translation(self,uv):
    # Converts the pixel coordinates to spatial coordinates WRT the camera's frame
    pose = self.arm_viewer.endpoint_pose()
    xy = [pose['position'].x,pose['position'].y,pose['position'].z]
    height, width, depth = self.frame #camera frame dimensions 
    print "\nx-pixel: ",uv[0],"\ty-pixel: ",uv[1]
    cx = uv[0] # x pixel of hand 
    cy = uv[1] # y pixel of hand
    pix_size = 0.0025 # Camera calibration (meter/pixels); Pixel size at 1 meter height of camera
    h = self.z_camera #Height from hand to camera, when at vision place
    x0b = xy[0] # x position of camera in baxter's base frame, when at vision place 
    y0b = xy[1] # y position of camera in baxter's base frame, when at vision place
    x_camera_offset = .045 #x camera offset from center of the gripper  
    y_camera_offset = -.01 #y camera offset from center of the gripper 
    #Position of the object in the baxter's stationary base frame
    x_camera = (cy - (height/2))*pix_size*h -0.045 # x-coordinate in camera's frame
    y_camera = (cx - (width/2))*pix_size*h  +0.01 # y-coordiante in camera's frame
    return x_camera,y_camera,h

  def xy_translation(self):
    uv = (self.x,self.y) # pixel coordinates
    if self.x != 0 and self.y != 0 and self.z_camera !=None: # If a hand has been detected and a position recorded
        x_camera,y_camera,h = self.pixel_translation(uv) # Obtains the spatial x,y coordinates
        self.depth_detection(float(x_camera),float(y_camera),h) # proceeds to coordinates translation
    else:
        self.depth_detection(0,0,0) #If a depth hasn't been detected, proceed to the depth detection 

  def depth_detection(self,x,y,z):
    pose = self.arm_viewer.endpoint_pose()
    pos = [pose['position'].x,pose['position'].y,pose['position'].z]
    quat = [pose['orientation'].x,pose['orientation'].y,pose['orientation'].z,pose['orientation'].w]

    height, width, depth = self.frame #camera frame dimensions 
    __matrix = self.tf.fromTranslationRotation(pos,quat) #self.orientation_cam
    # If a depth has already been detected
    # Proceed to coordinates translation, and use inverse kinematics to move to position
    
    if self.z_camera != None:
        print "depth detected"
        z = self.z_camera 
        xyz = np.dot(__matrix,np.matrix([[x],[y],[z],[1]]))
        print "\nx-coordinate obtained: ",xyz[0],"\ny-coordinate obtained: ",xyz[1],"\nz-coordinate obtained: ",xyz[2]
        pos = [xyz[0],xyz[1],xyz[2]]
        self.ik(self.limb,pos,self.orientation_shaker)
    # Else, move the arm towards unit vector until a depth has been detected
    else:
    
    print "...Moving arm..."
    __matrix = __matrix[:3,:3]
    # Aligns the end effector to the detected hand before moving it
    n = self.unit_vector_to_point(self.x,self.y)
    if self.x > width/2: # If self.x > widht/2, hand is above the camera
        up = True # Will pass this as an argument to generate the unit vector, indicating that the hand is above
    else: up = False
    u_vector_gripper = self.unit_vector_gripper_frame(n)
    u_vector = np.dot(__matrix,np.matrix([[u_vector_gripper[0]],[u_vector_gripper[1]],[u_vector_gripper[2]]]))
    print "...Aligning arm..."
    self.align(u_vector)
    #self.with_check(__matrix,up)
    self.without_check(__matrix,up)

  def without_check(self,__matrix,up = False):
    height, width, depth = self.frame #camera frame dimensions 
    while self.z_camera == None and self.torque<20: # While no depth is detcted and no torque is detected
        n = self.unit_vector_to_point(width/2,height/2)# Unit vector to the center of the camera
        # the unit vector to the center of the camera is generated, since after alignement, the hand should be almost in line with the center
        u_vector_gripper = self.unit_vector_gripper_frame(n) # converts the unit vector to the gripper's frame
        u_vector = np.dot(__matrix,np.matrix([[u_vector_gripper[0]],[u_vector_gripper[1]],[u_vector_gripper[2]]])) # converts to the base frame
        if self.limb == 'left':
            if up == True: # Accounts for the position of the hand, above or below the camera
                # Unit vector seems to always have a negative z-component, hence if the hand is detected above the camera, the z-component is negated
                u_vector[2] = -u_vector[2] # negating the z-component
        elif self.limb == 'right':
            if up != True: # The inverse is true for the right arm
                u_vector[2] = -u_vector[2]   
        pose = self.arm_viewer.endpoint_pose()        
        pos = [pose['position'].x+(u_vector[0]/20),pose['position'].y+(u_vector[1]/20),pose['position'].z+(u_vector[2]/20)] # Move the arm by increments
        quat = [pose['orientation'].x,pose['orientation'].y,pose['orientation'].z,pose['orientation'].w]
        self.ik(self.limb,pos,quat)
    # Once depth has been detected, change orientation of gripper to that of the shaker, defined when initialized
    pose = self.arm_viewer.endpoint_pose()
    pos = [pose['position'].x,pose['position'].y,pose['position'].z]
    if self.limb == 'left':
        self.ik('left',pos,self.left_orientation_shaker,True) # Hand-shaking position for left arm
    else:
        self.ik('right',pos,self.right_orientation_shaker,True) # Hand-shaking position for right arm
    print "Position reached...Moving to Instructor position"
    #self.follow_up() # Follow_up instructions

  def with_check(self,__matrix,up):
    height, width, depth = self.frame #camera frame dimensions 
    print self.hand_area
    while self.z_camera == None and self.torque<20 and self.hand_area>5000: # As long as hand area is detected
        if self.hand_area > 5000:
            n = self.unit_vector_to_point(width/2,height/2)
            u_vector_gripper = self.unit_vector_gripper_frame(n)
            u_vector = np.dot(__matrix,np.matrix([[u_vector_gripper[0]],[u_vector_gripper[1]],[u_vector_gripper[2]]]))
            if self.limb == 'left':
                if up == True:
                    u_vector[2] = -u_vector[2]
            elif self.limb == 'right':
                print self.limb
                if up != True:
                    u_vector[2] = -u_vector[2]   
            pose = self.arm_viewer.endpoint_pose()        
            pos = [pose['position'].x+(u_vector[0]/20),pose['position'].y+(u_vector[1]/20),pose['position'].z+(u_vector[2]/20)]
            quat = [pose['orientation'].x,pose['orientation'].y,pose['orientation'].z,pose['orientation'].w]
            self.ik(self.limb,pos,quat)
        else:
            print "No hands detected..."
            return 0

    if self.hand_area > 5000:
        pose = self.arm_viewer.endpoint_pose()
        pos = [pose['position'].x,pose['position'].y,pose['position'].z]
        self.ik('left',pos,self.left_orientation_shaker,True)
        print "Position reached...Moving to Instructor position"
        #self.follow_up()
    else:
        print "No hands detected for final alignment"

  def ik(self,limb,pos,quat,block=None,arm=None):
    '''
    This function uses inverse kinematics to calculate the joint states given a certain pose.
    It also applies the joint states to the specified arms, i.e moves the arm. Arguments:
    - limb : the limb which is to be moved. If not specified, limb is viewer limb
    - pos,quat : pose for which joint solutions are desired
    - block - if block is None, the motion of the joints will be done by set_joint_positions. 
    Else, it will be done by move_to_joint_positions.
    set_joint_positions allows the operation to be interupted (Used when moving the arm towards the hand)
    move_to_joint_positions cannot be interupted, and is used when aligning the end effector

    ******* CORE OF FUNCTION IS FROM /BAXTER_EXAMPLES/IK_SERVICE_CLIENT.PY********
    '''
    if arm == None:
        arm = self.arm_viewer

    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        'left': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                ),
                orientation=Quaternion(
                    x=quat[0],
                    y=quat[1],
                    z=quat[2],
                    w=quat[3],
                ),
            ),
        ),
        'right': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                ),
                orientation=Quaternion(
                    x=quat[0],
                    y=quat[1],
                    z=quat[2],
                    w=quat[3],
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
        #print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
        #      (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        # reformat the solution arrays into a dictionary
        joint_solution = dict(zip(resp.joints[0].name, resp.joints[0].position))
        if block == None:
            arm.set_joint_positions(joint_solution)
        else: 
            arm.move_to_joint_positions(joint_solution)
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")
    return 0

def main():
  while True:
    limb = raw_input("Which will be the moving limb? (l/r) ")
    if limb == 'l' or limb == 'r':
        if limb == 'l':
            limb = 'left'
            other_limb= 'right'
        else:
            limb = 'right'
            other_limb = 'left'
        break
  parameter_reset = raw_input("Reset camera parameters?(y/n)")
  querry = raw_input("Is the arm at the correct starting position?(y/n) ")
  if querry == 'y':
    rospy.init_node('get_pos', anonymous=True)
    if parameter_reset == 'y':
        position = get_pos(limb,other_limb,True)
    else: 
        position = get_pos(limb,other_limb,None)
    rospy.sleep(5) # Run the hand detection for a few seconds to get a constant pair of coordinates
    position.xy_translation() # Get the spatial coordinates of the detected hand
  elif querry == 't':
    rospy.init_node('get_pos', anonymous=True)
    if parameter_reset == 'y':
        position = get_pos(limb,other_limb,True)
    else: 
        position = get_pos(limb,other_limb,None)
    rospy.spin()
  else:
    rospy.init_node('get_pos', anonymous=True)
    if parameter_reset == 'y':
        position = get_pos(limb,other_limb,True)
    else: 
        position = get_pos(limb,other_limb,None)
    position.get_starting_pos()

if __name__ == '__main__':
    main()
