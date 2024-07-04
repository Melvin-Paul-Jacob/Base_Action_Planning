import rospy
from test_pipe_ros import Pipe
import argparse
import detector_yolov7 as dy7
from Nav_Man import Starter
import sys

rospy.init_node('image_converter')
#rospy.init_node('Mover')
#rospy.init_node('Starter')

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='Object_Detection_yolov7/inference/saved', help='source')  # file/folder, '0' for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, default=[39,40,41,42,43,44,45,46], help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
opt = parser.parse_args()

yd = dy7.YoloDetector(opt)

try:
    pipeline = Pipe()
    pipeline.gesture_object_package(yd)
    #pipeline.plan_package()
    sys.exit()
except KeyboardInterrupt:
    sys.exit()