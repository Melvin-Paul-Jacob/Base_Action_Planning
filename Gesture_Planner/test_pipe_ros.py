import cv2
import GestDetClass as gdc
from Load_stream import image_converter
from Nav_Man import Mover
from t2D23D import t2d2t3d
import open3d as o3d
import numpy as np
from collections import deque 
import warnings
import time
import sys
import rospy
warnings.filterwarnings("ignore")

class Pipe():
    def __init__(self):
        self.g_real_object_pose = None
        self.g_valid_points = []
        self.cam = image_converter()
        self.mover = Mover()
        cv2.destroyAllWindows()
        self.handdetector = gdc.GestDetectorClassifier(classifier_path='models/gesture_model_key2.tflite',gest_thresh=60)
        self.td23D = t2d2t3d("map.pgm") 
        self.found_boxes = []
        self.pose=None
        self.prev_selection=None
        self.cam.img_publisher()

    def gesture_object_package(self, yd):
        try:
            while True:
                frame, cloud = self.cam.get_frame()
                img= frame.copy()
                image = frame.copy()
                img, gest, selected_box = self.handdetector.gest_action(img, self.found_boxes)
                #print(gest)
                self.cam.publish_img = img
                if gest=="scan":
                    self.handdetector.gest_queue = deque(maxlen=30)
                    self.found_boxes = yd.get_bound(image)
                if len(selected_box)>0 and self.prev_selection!=selected_box:
                    self.cam.publish_img = []
                    try:
                        #time.sleep(3)
                        #cloud = self.cam.get_cloud()
                        whole, obj_clus = self.td23D.get_box_voxel(selected_box, cloud)
                        mean_coords = obj_clus.get_center()
                        mea = self.td23D.show_point(mean_coords,col=[0,1,0])
                        aabb = obj_clus.get_oriented_bounding_box()
                        o3d.visualization.draw_geometries([whole, obj_clus, mea, aabb])
                        obj_pose = self.td23D.get_3D_cords(obj_clus) #open3d frame
                    except Exception as e:
                        print(e)
                        self.prev_selection = None
                        selected_box = []
                        print("rescan")
                        continue
                    dist = ((obj_pose.pose.position.x**2)+(obj_pose.pose.position.y**2))**0.5
                    base_pose_robot = self.mover.rob_cur_cor
                    print(dist)
                    if dist > 0.5:
                        print("---------------object is far----------")
                        valid_points_3D = self.td23D.get_valid_coords_3D(obj_pose,whole,obj_clus,min_radius=0.5,max_radius=0.9,num_points=15,padding=0.3,viz=True) #open3d frame
                        valid_points_head = self.mover.transform_3D2head(valid_points_3D)
                        real_object_head = self.mover.transform_3D2head([obj_pose])[0]
                        valid_points_map = self.mover.transform_head2map(valid_points_head)
                        real_object_pose = self.mover.transform_head2map([real_object_head])[0]
                        #sys.exit()
                        if len(valid_points_map)==0:
                            print("cannot pickup object")
                            self.prev_selection = None
                            selected_box = []
                            continue
                        valid_points = self.td23D.get_valid_coords_2D(valid_points_map,base_pose_robot,real_object_pose,viz=True)
                        #print(rob_cords)
                        if len(valid_points)==0:
                            print("cannot pickup object")
                            self.prev_selection = None
                            selected_box = []
                            continue
                        valid_pose_array = self.td23D.list2posearray(valid_points)
                        if len(valid_pose_array.poses)>0:
                            self.g_real_object_pose = real_object_pose
                            self.g_valid_points = valid_pose_array
                            break
                
                #cv2.imshow("Image", img)
                '''k = cv2.waitKey(1) & 0xff
                if k == 27:
                    print("-> Ending Video Stream")
                    #self.cam.release_camera()
                    cv2.destroyAllWindows()
                    break  '''
            #self.cam.publish_img = []
            #self.cam.release_camera()
            #cv2.destroyAllWindows()
        except KeyboardInterrupt:
            sys.exit()

    def plan_package(self):
        try:
            base_pose_robot = self.mover.rob_cur_cor
            print(self.g_real_object_pose)
            valid_points = self.td23D.order_valid(self.g_valid_points, base_pose_robot, self.g_real_object_pose)
            print(valid_points)
            self.mover.visualize_rviz(valid_points,self.g_real_object_pose,base_pose_robot)
            # print("Moving")
            #del valid_points.poses[idx]
            self.mover.pipeline_pub.publish(valid_points)
            print("Published valid pose")
            self.mover.pipeline_pub_obj.publish(self.g_real_object_pose)
            print("Published object pose")
            sys.exit()
        except KeyboardInterrupt:
            sys.exit()

'''self.mover.go_to_loc(ordered_point)
                #timer code
                manipulate = self.mover.check_goal_reached()
                print("manipulate",manipulate)
                if manipulate == 0:
                    print("service restart started")
                    # with open('restart.txt', 'w') as f:
                    #     f.write('True')
                    #     f.close()
                    # goalid = GoalID()
                    # goal_id.id = ""
                    # self.mover.cancel_goals_pub.publish(goal_id)
                    # subprocess.call('start /wait python service_restarter.py', shell=True)
                    # subprocess.run(["bash", "call_services_heartmet", "restart"], shell=True)
                    print("Services restart complete")
                    time.sleep(1)
                    continue
                elif manipulate == 2:
                    time.sleep(2)
                    continue
                else:
                    #get new relative object pose type here
                    #initial_obj- rob_moved
                    new_obj_pose = copy.deepcopy(real_obj_pose)
                    new_obj_pose.pose.position.x -= ordered_point.pose.position.x
                    new_obj_pose.pose.position.y -= ordered_point.pose.position.y
                    print("------------------grabing-------------------")
                    #transformed_pose = mover.transform_head2base(new_obj_pose)
                    #self.mover.grab(transformed_pose)
                    #self.mover.check_manipulation_completed()
                    # marker = Marker()
                    break'''
