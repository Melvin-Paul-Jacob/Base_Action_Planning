import argparse
from pathlib import Path
import cv2
import torch
from numpy import random
import sys
sys.path.insert(0, './Object_Detection_yolov7')
from Object_Detection_yolov7.models.experimental import attempt_load
from Object_Detection_yolov7.utils.datasets import LoadImages
from Object_Detection_yolov7.utils.general import check_img_size, non_max_suppression, \
    scale_coords, strip_optimizer
from Object_Detection_yolov7.utils.plots import plot_one_box
from Object_Detection_yolov7.utils.torch_utils import select_device, TracedModel

class YoloDetector():
    def __init__(self,opt):
        self.source, self.view_img, self.imgsz, self.trace = opt.source, opt.view_img, opt.img_size, not opt.no_trace
        self.opt=opt
        self.weights = 'yolov7-tiny.pt'
        self.device = select_device('cpu')

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, self.device, 640)
            
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            
    def get_bound(self,image):
        cv2.imwrite(self.source+"/det.png",image) 
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        with torch.no_grad():
            if self.opt.update:  # update all models (to fix SourceChangeWarning)
                rects = self._detect(dataset)
                return rects
                strip_optimizer(self.weights)
            else:
                rects = self._detect(dataset)
                return rects

    def _detect(self, dataset):
        rects = []
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.opt.augment)[0]
    
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
    
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        rect = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]
                        rects.append(rect)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    cv2.imwrite(str(p),im0)
        return rects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='Object_Detection_yolov7/inference/saved', help='source')  # file/folder, '0' for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0,39,40,41,42,43,44,45,46], help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    yd = YoloDetector(opt)
    print(yd.get_bound())
    

    
