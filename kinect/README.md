# Baxter Lab Reaching Hand Kinect

This package contains an actionlib.  When executed, the actionlib will wait for the kinect to detect a human hand, and will then move Baxter's hand to that human hand.  If the human hand moves, Baxter will dynamically change it's target.  The actionlib's feedback indicates how far Baxter's arm is from the target.  This actionlib is heavily configureable using **config.yaml**, which has a description of each parameter.  NOTE: **HandDetector.py** can also be run as a separate node, which merely uses the kinect to detect a hand and publish it's location to a topic.

### Sample Usage
1. Make sure the kinect is plugged in to the laptop, and start the kinect drivers by running **roslaunch openni_launch openni.launch**
2. Execute the action lib launch file, by running **roslaunch lab_reaching_hand_kinect reaching_hand.launch**.  NOTE: This currently also starts a sample client.  If you would like to use your own client, comment that line out of the launch file.

### Questions?
Contact <amaln@cmu.edu>
