import os
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import pandas as pd
import itertools
from collections import deque 

class GestDetectorClassifier():
    def __init__(self, classifier_path, gest_thresh=60):
        classifiername, classifier_extension = os.path.splitext(classifier_path)
        if classifier_extension==".tflite":
            self.interpreter = tf.lite.Interpreter(model_path=classifier_path,num_threads=1)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.lite=True
        else:
            self.model = load_model(classifier_path)
            self.lite=False
        self.gest_thresh=gest_thresh
        self.classify_size=32
        self.maxHands = 1
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(False, 1, 1, 0.6, 0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.gestures = ["stop", "scan", "point", "others"]  
        self.min_frames=60
        self.min_time = 3
        self.gest_queue = deque(maxlen=30)
        self.point_queue = deque(maxlen=30)
        self.selected_box=[]
        self.cur_gesture="None"
        self.prev_gesture="None1"
        self.gest_start = 0
        self.gest_end = 0
        
    def gesture_key_classify(self, lmLists):
        if len(lmLists)>0:
            X=[]
            for lmList in lmLists:
                lmList = lmList-lmList[0]
                l1 = list(itertools.chain.from_iterable(lmList[:,1:]))
                max_value = np.max([abs(x) for x in l1])
                landmark_list = l1/max_value
                X.append(landmark_list)
            X = np.array(X, dtype=np.float32)
            if self.lite:
                input_details_tensor_index = self.input_details[0]['index']
                self.interpreter.set_tensor(input_details_tensor_index,X)
                self.interpreter.invoke()
                output_details_tensor_index = self.output_details[0]['index']
                preds = self.interpreter.get_tensor(output_details_tensor_index)
            else:
                preds = self.model.predict(X,verbose=0)
            probabilityValues = np.amax(preds,axis=1)*100
            prob = probabilityValues.astype(int)
            pred = np.argmax(preds,axis=1)
            out = np.vstack((pred,prob)).T
            return out
        else:
            return []
    
    def findHands(self, img, draw_box=True, draw_key=False):
        boxes = []
        lmLists = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        h, w = img.shape[:2]
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks is not None:
            for handLms in self.results.multi_hand_landmarks:
                bbox = []
                landmarks_array = np.array([[i,int(lm.x*w), int(lm.y*h)] for i, lm in enumerate(handLms.landmark)])
                xmin, ymin, hw, hh = cv2.boundingRect(np.array(landmarks_array[:,1:]))
                xmax, ymax = xmin+hw, ymin+hh
                xmin, ymin, xmax, ymax = xmin-20, ymin-20, xmax+20, ymax+20
                bbox = [xmin, ymin, xmax, ymax]
                # Filter based on size
                area = hw*hh
                #print(area)
                if self.classify_size**2 < area:
                    lmLists.append(landmarks_array)
                    boxes.append(bbox)
                    if draw_key:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    if draw_box:
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),(0, 255, 0), 2)
                if len(lmLists)==self.maxHands:
                    break
        return img, lmLists, boxes
    
    def get_index_line(self, img, lmList):
        height, image_w = img.shape[:2]
        val = []
        try:
            if len(lmList) != 0:
                ind_base = lmList[5]
                ind_tip = lmList[8]
                if (ind_base[1]>=0 and ind_base[2]>=0 and ind_base[1]<image_w and ind_base[2]<height and 
                ind_tip[1]>=0 and ind_tip[2]>=0 and ind_tip[1]<image_w and ind_tip[2]<height):
                    if ind_tip[1]==ind_base[1] and ind_tip[2]!=ind_base[2]:
                        slope = np.deg2rad(90)
                    elif ind_tip[1]!=ind_base[1] and ind_tip[2]==ind_base[2]:
                        slope = np.deg2rad(0)
                    elif ind_tip[1]==ind_base[1] and ind_tip[2]==ind_base[2]:
                        slope = np.deg2rad(0)
                    else:
                        slope = (ind_tip[2]-ind_base[2])/(ind_tip[1]-ind_base[1])
                    intercept = ind_tip[2] - (ind_tip[1]*slope)
                    if ind_base[1]>=ind_tip[1]:
                        linexs = np.arange(0, ind_tip[1])
                        lineps =  [[x, int(slope*x + intercept)] for x in linexs if int(slope*x + intercept)>=0]
                        end = tuple(lineps[0])
                    else:
                        linexs = np.arange(ind_tip[1], image_w)
                        lineps =  [[x, int(slope*x + intercept)] for x in linexs if int(slope*x + intercept)>=0]
                        end = tuple(lineps[-1])
                    start = (ind_tip[1],ind_tip[2])
                    val = [start, end, lineps]
        except: 
            pass
        return val
    
    def intersect_box(self, img, lmList, rects, draw_line=True):
        val = self.get_index_line(img, lmList)
        intersecting = []
        if len(val)!= 0:
            for i in range(len(rects)):
                ll = rects[i][0]
                ur = rects[i][1]
                inters = []
                for p in val[2]:
                    if ll[0]<=p[0]<=ur[0] and ll[1]<=p[1]<=ur[1]:
                        inters.append(p)
                #print(inters)
                if len(inters)>0:
                    dists = [math.dist(list(inter),list(val[0])) for inter in inters]
                    dists = np.array(dists)
                    ind = np.argsort(dists)[0]
                    intersecting.append([i,inters[ind]])
        if len(intersecting)==0:
            if draw_line and len(val)!= 0:
                cv2.line(img, val[0],val[1], (255,0,0), 2)
            return img, None
        else:
            dists = []
            for i in intersecting:
                center = i[1]
                center_dist = math.dist(list(center),list(val[0]))
                dists.append(center_dist)
            dists = np.array(dists)
            inter_ind = np.argsort(dists)[0]
            ind = intersecting[inter_ind][0]
            if draw_line:
                cv2.line(img, val[0],tuple(intersecting[inter_ind][1]), (255,0,0), 2)
                cv2.circle(img,tuple(intersecting[inter_ind][1]),4, (0,255,224),-1)
            return img, ind
        
    def gest_action(self, img, found_boxes=[],draw_box=1, draw_key=1):
        gest="None"
        #image = img.copy()
        for i in range(len(found_boxes)):
            cv2.rectangle(img, found_boxes[i][0], found_boxes[i][1], (0,0,255), 2)
        img, lmLists, boxes = self.findHands(img, draw_box=draw_box, draw_key=draw_key)
        hands = self.gesture_key_classify(lmLists)
        if len(hands)==0:
            self.gest_start = time.time()
            self.gest_queue = deque(maxlen=30)
            self.point_queue = deque(maxlen=30)
        for i, hand in enumerate(hands):
            index, prob = hand
            x1,y1,x2,y2 = boxes[i]
            gesture = self.gestures[index]
            #print(self.gest_queue)
            if prob>self.gest_thresh:
                self.prev_gesture = self.cur_gesture
                self.cur_gesture = gesture
                if self.prev_gesture!=self.cur_gesture:
                    self.gest_start = time.time()
                    self.gest_queue = deque(maxlen=30)
                    self.point_queue = deque(maxlen=30)
                else:
                    self.gest_end = time.time()
                self.gest_queue.append(gesture)
                cv2.putText(img, gesture+" "+str(prob)+"%", (x1+10,y1-20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),2)
                if gesture=="point" and len(found_boxes)!=0:
                    img, rect_ind = self.intersect_box(img, lmLists[i], found_boxes)
                    if rect_ind!=None:
                        self.point_queue.append(rect_ind)
                    else:
                        self.point_queue = []
                if len(list(set(self.gest_queue)))==1 and (self.gest_end-self.gest_start)>=self.min_time:
                    gest = list(set(self.gest_queue))[0]
                    if gest=="point" and len(list(set(self.point_queue)))==1:
                        rect_indi = list(set(self.point_queue))[0]
                        self.selected_box = found_boxes[rect_indi]
                    elif gest=="scan":# and len(found_boxes)==0:
                        #found_boxes = objectdetector.get_bound(image)
                        pass
                        #print(found_boxes)
                    elif gest=="stop":
                        self.selected_box=[]
                    self.gest_start = time.time()
                    self.gest_queue = deque(maxlen=30)
                    self.point_queue = deque(maxlen=30)
                else:
                    gest = "None"
            else:
                self.gest_queue.append("others")
                gesture="others"
                self.prev_gesture = self.cur_gesture
                self.cur_gesture = gesture
        if len(self.selected_box)>0:
            cv2.rectangle(img, self.selected_box[0], self.selected_box[1], (0,255,0), 2)
        return img, gest, self.selected_box
    
                
def main():
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = GestDetectorClassifier(classifier_path='models/gesture_model_key.tflite')
    while True:
        success, img = cap.read()
        if success == False:
            print('-> Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            cv2.destroyAllWindows()
            cap.release()
            break
        #img = cv2.flip(img, 1)
        img, lmLists, boxes= detector.findHands(img)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            print("-> Ending Video Stream")
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()