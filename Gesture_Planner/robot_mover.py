#!/usr/bin/env python3

from math import pi
import sys
import os
import tf
from geometry_msgs.msg import TwistStamped, PoseStamped, Twist
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, os
import rospy
import roslib
import tf2_ros
from mas_hsr_gripper_controller.gripper_controller import GripperController
import math
import geometry_msgs.msg
from copy import deepcopy

class ArmBase:
    def __init__(self):
        self.arm = MoveGroupCommander("arm", wait_for_servers=0.0)
        self.arm.set_max_acceleration_scaling_factor(0.4)
        self.gripper_controller = GripperController()
        self.loop_rate = rospy.Rate(20)
        self.loop_rate.sleep()
        self.tfbuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfbuffer)
        self.tf_listener = tf.TransformListener()
        self.Target_Active = False  
        print("Optimum motion planning Server started")

        # in m or m/sec
        self.tol = 0.004
        self.arm_vel = 0.01
        self.arm_offset = 0 #-0.08
        self.tip_offest = 0 #0.07
        # self.go_to_home()

    def get_arm_cur(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:
                trans = self.tfbuffer.lookup_transform('base_link', 'hand_palm_link', rospy.Time())
                # print(trans)
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                pass
        return trans

    def distance(self, a, b):
        if(a == b):
            return 0
        elif (a < 0) and (b < 0) or (a > 0) and (b > 0):
            if (a < b):
                return (abs(abs(a) - abs(b)))
            else:
                return -(abs(abs(a) - abs(b)))
        else:
            return math.copysign((abs(a) + abs(b)),b)

    def go_to_home(self):
        arm_target = {'arm_lift_joint': 0.0,
                      'arm_flex_joint': 0.0,
                      'arm_roll_joint': 0.0,
                      'wrist_flex_joint': -pi/2.,
                      'wrist_roll_joint': 0.,
                      'wrist_ft_sensor_frame_joint': 0.0}
        
        self.arm.set_joint_value_target(arm_target)
        ax = self.arm.go()
        self.loop_rate.sleep()

        print("initial arm pose:")
        arm_pos = self.get_arm_cur()
        print(arm_pos.transform.translation.x)
        print(arm_pos.transform.translation.y)
        print(arm_pos.transform.translation.z)    

    def pick_obj(self):
        print("Closing gripper")
        self.gripper_controller.close()
        self.loop_rate.sleep()

    def place_obj(self):
        print("Opening gripper")
        self.gripper_controller.open()
        self.loop_rate.sleep()
            
    def move_arm(self, xs=0.0,ys=0.0,zs=0.0):
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time().now()
        msg.twist.linear.x = xs
        msg.twist.linear.y = ys
        msg.twist.linear.z = zs
        self.cartesian_vel_pub.publish(msg)
        self.loop_rate.sleep()
        
    def obj_pose(self, objec_pose):
        PoseTfScuucess = False
        self.go_to_home()

        arm_pos = self.get_arm_cur()
        arm_init_x = arm_pos.transform.translation.x
        arm_init_z = arm_pos.transform.translation.z

        if self.Target_Active == False and PoseTfScuucess == True:
            self.Target_Active = True
            objec_pose.pose.position.y -= self.arm_offset
            objec_pose.pose.position.x += self.tip_offest
            
            x_to_move = objec_pose.pose.position.x
            z_to_move = objec_pose.pose.position.z

            #Z axis Up +ve & Down -ve
            z_relative_move = z_to_move - arm_init_z
            if (z_to_move > 0) and (z_relative_move > 0):
                print("Moving Z forward")
                while  not (z_relative_move-self.tol < abs(self.distance(self.get_arm_cur().transform.translation.z, arm_init_z)) < z_relative_move+self.tol):
                    self.move_arm(zs=self.arm_vel)
                print("z motion complete")
            else:
                print("Wrong Z target pose")
                return 0
            self.move_arm()

            #X axis Forward +ve and Backward -ve 
            x_relative_move = x_to_move - arm_init_x
            if (x_to_move > 0) and (x_relative_move > 0):
                print("Moving X forward")
                while  not (x_to_move-self.tol < abs(self.distance(self.get_arm_cur().transform.translation.x, arm_init_x)) < x_to_move+self.tol):
                    print(x_to_move, abs(self.distance(self.get_arm_cur().transform.translation.x, arm_init_x)))
                    self.move_arm(xs=self.arm_vel)
                self.move_arm()
                print("x motion complete")
            else:
                print("Wrong X target pose")
                return 0

            self.Target_Active = False
            self.pick_obj()
            self.place_obj()
            self.go_to_home()
            return 1