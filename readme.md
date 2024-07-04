WS22 - Gesture Recognition

The aim of the project is to identify the gesture shown by human and tracing an optimal path to grab the object for pickup gesture.

<img width="537" alt="gesture" src="https://user-images.githubusercontent.com/86923487/223169041-73e0dca5-c0d4-48bb-bfac-fda09423cc2b.PNG">

# Table of Contents:
1. [Contributions](#contributions)
2. [Introduction](#introduction)
3. [Architecture](#architecture)
4. [Organisation](#organisation)
5. [Dependencies](#dependcies)
6. [Things to consider](#things)
7. [Getting started](#getting)
8. [Acknowledgement](#ack)


## Contributions: <a name="contributions"></a>
**Gesture Recognition**
* Faster and more accurate gesture recognition
* Introduced invariance to background and lighting conditions
* Improved accuracy of pointing feature

**3D Object Detection**
* Migration from PCL to Open3D
* Improved detection by enclosing object in a 3D bounding box
* Removed noise using outlier removal
* Used clustering to segment out the object
* Introduced 3D visualization
* Added Rviz markers

**Optimal Grab Pose Estimator**
* Ensures minimum distance from obstacle
* Ensures that the arm is not over extended
* Produce a fixed no. of poses around the object for optimal grab action
* Check for obstacle in the path of the arm before manipulation
* Picking up object from the closest valid location

**Localisation**
* Fixed localisation issue by publishing the map to localiser


## Introduction: <a name="introduction"></a>
The robot identifies the following list of gestures shown by human and perform the respective action.
* Scan - Perform object detection and generate bounding boxes for the list of COCO Dataset objects in the current robot vision scene
* Point - Select the object pointed by human
* Stop - Erase the bounding boxes generated in the robot vision scene

Once the launch file is triggered, the scan gesture is shown by human and the robot generates bounding box for the objects in the scene. The human then points the hand at one of the selected object. The pose conversion from 2D to 3D is done for the selected object. Ten possible target poses for the robot base is generated. The point cloud path validation for all base target poses are performed with respect to the object pose, to remove the invalid base target poses. Then navigation starts with one of the valid target base pose and if the robot is not able to plan the path in 10 seconds, the next valid pose is attempted. Once the robot reaches the target base pose, manipulation is performed to grab the object. 

![image](https://user-images.githubusercontent.com/86923487/223179411-89932b07-43dd-4a48-9a96-f97bfcfd0e9e.png)


## Architecture <a name="architecture"></a>
![SDP](https://user-images.githubusercontent.com/86923487/224802698-9e14882b-843c-4b05-b1d9-99164da46f6f.jpg)
![flow](https://user-images.githubusercontent.com/102292470/224801154-59dc10c2-7b43-4b02-9c80-f4803d1fa2c2.jpg)


## Organisation: <a name="organisation"></a>
**gesture_detect_classify** metapackage contains the following packages
* <code>test_pipe_ros</code> - Launch file, optimal path planner
  * <code>GestDetClass</code> - Gesture recognition inferencing package
  * <code>detector_yolov7</code> - Yolo object detection package
  * <code>t2d2t3d</code> - Get 3D pose of the selected object, obstacle detection
  * <code>Load_stream</code> - Navigation, Manipulation, Validation


## Dependencies: <a name="dependcies"></a>
* mdr_knowledge_base
* mdr_object_recognition
* mdr_cloud_object_detection
* mdr_perception_msgs
* mdr_manipulation_msgs
* mdr_manipulation 
* mdr_perception 
* mdr_navigation 

## Things to consider: <a name="things"></a>
1. You may need to restart services on the robot.
   * hsr_move_arm_action.service
   * hsr_move_arm_joints_action.service
   * hsr_move_base_action.service
   * hsr_move_base_client.service
   * hsr_move_forward_action.service
   * hsr_perceive_plane_action.service
   * hsr_perceive_plane_client.service
   * hsr_pickup_action.service
   * hsr_pickup_client.service
   
2. Localize the robot accurately

## Getting Started: <a name="getting"></a>
1. pip3 install open3d
2. pip3 install tensorflow
3. pip3 install mediapipe
4. mkdir ~/catkin_ws/src
5. cd ~/catkin_ws/src
6. git clone <code>https://github.com/b-it-bots/mas_domestic_robotics.git</code>
7. git clone <code>https://github.com/HBRS-SDP/ws22-gesture_recognition.git</code>
8. cd ~/catkin_ws/
9. catkin build
10. cd ~/catkin_ws/
11. python3 ws22-gesture_recognition/gesture_detect_classify_new/test_pipe_ros.py


## Acknowledgements: <a name="ack"></a>
* Thanks to all <code>b-it-bots mas_domestic_robotics</code> [contributors](https://github.com/b-it-bots/mas_domestic_robotics/graphs/contributors)
