import argparse
from pathlib import Path

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel


def detect(dataset):
    rects = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

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
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    rect = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]
                    rects.append(rect)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                cv2.imwrite(str(p),im0)
    return rects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/saved', help='source')  # file/folder, '0' for webcam
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
    source, view_img, imgsz, trace = opt.source, opt.view_img, opt.img_size, not opt.no_trace
    weights = 'yolov7-tiny.pt'
    device = select_device('cpu')

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    cap=cv2.VideoCapture(0+ cv2.CAP_DSHOW)
    # We need to check if camera is opened previously or not 
    if (cap.isOpened() == False):  
        print("-> Error reading video file") 
    i=0   
    #capture 60th frame //not necessary can also be the 1st
    while(1):
        #cv2.waitKey(1000)
        ret,img = cap.read()
        i+=1
        if(i==60 and ret):
            cv2.imwrite(source+"/"+str(i)+".png",img) 
            break
    cap.release()
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            rects1 = detect(dataset)
            print(rects1)
            strip_optimizer(weights)
        else:
            rects1 = detect(dataset)
            print(rects1)