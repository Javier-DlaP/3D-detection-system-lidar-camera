import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.autograd import Variable
from torchvision import transforms

#Functions to operate with kitti dataset

def read_params(file_path):

    camera_param = open(file_path,'r')
    matrix = camera_param.readlines()
    camera_param.close()
    matrix = (matrix[2].split(':'))[1].split(' ')
    matrix.pop(0)
    matrix[11] = matrix[11].rstrip('\n')
    matrix = [float(i) for i in matrix]
    
    p = np.vstack((matrix[0:4],matrix[4:8],matrix[8:12]))
    return p


def print2D_centers(imgL,label_path,p):
    labels = open(label_path,'r')
    for label in labels:
        field = label.split(' ')
        if field[0] != 'DontCare':
            point1 = (int(float(field[4])),int(float(field[5])))
            point2 = (int(float(field[6])),int(float(field[7])))
            cv2.rectangle(imgL,point1,point2,(0,255,0),1)
            center = field[11:14]
            center = [float(i) for i in center]
            center.append(1)
            center = np.transpose(np.array(center))
            center = np.matmul(p,center)
            cv2.drawMarker(imgL,(int(center[0]/center[2]),int(center[1]/center[2])),(0,0,255))
    labels.close()

def compute_draw_3D(imgL,label_path,p):
    labels = open(label_path,'r')
    for label in labels:
        field = label.split(' ')
        if field[0] != 'DontCare':
            #Create rotation matrix
            rotation = float(field[14])
            sin = math.sin(rotation)
            cos = math.cos(rotation)
            rot_mat = np.vstack(([cos,0,sin],[0,1,0],[-sin,0,cos]))
            #Create de 3D bbox in real camera coordenates
            l = float(field[10])
            w = float(field[9])
            h = float(field[8])  
            center = field[11:14]
            center = [float(i) for i in center]            
            cube = np.vstack(([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2],[0,0,0,0,-h,-h,-h,-h],[w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]))
            offset = np.vstack((np.full((1,8),center[0]),np.full((1,8),center[1]),np.full((1,8),center[2])))
            cube3D = np.matmul(rot_mat,cube) + offset
            cube3D = np.vstack((cube3D,np.full((1,8),1)))
            #Transform to image coordenates and plot
            cube_image = np.matmul(p,cube3D)

            for i in range(8):
                cube_image[0,i] = cube_image[0,i]/cube_image[2,i]
                cube_image[1,i] = cube_image[1,i]/cube_image[2,i]
            
            #Draw points
            #for i in range(8):
            #    cv2.drawMarker(imgL,(int(cube_image[0,i]),int(cube_image[1,i])),(0,0,255))
            #Draw cube

            cv2.line(imgL,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,1]),int(cube_image[1,1])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,3]),int(cube_image[1,3])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,2]),int(cube_image[1,2])),(int(cube_image[0,1]),int(cube_image[1,1])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,2]),int(cube_image[1,2])),(int(cube_image[0,3]),int(cube_image[1,3])),(0,0,255),1)

            cv2.line(imgL,(int(cube_image[0,4]),int(cube_image[1,4])),(int(cube_image[0,5]),int(cube_image[1,5])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,4]),int(cube_image[1,4])),(int(cube_image[0,7]),int(cube_image[1,7])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,6]),int(cube_image[1,6])),(int(cube_image[0,5]),int(cube_image[1,5])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,6]),int(cube_image[1,6])),(int(cube_image[0,7]),int(cube_image[1,7])),(0,0,255),1)

            cv2.line(imgL,(int(cube_image[0,0]),int(cube_image[1,0])),(int(cube_image[0,4]),int(cube_image[1,5])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,1]),int(cube_image[1,1])),(int(cube_image[0,5]),int(cube_image[1,5])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,2]),int(cube_image[1,2])),(int(cube_image[0,6]),int(cube_image[1,6])),(0,0,255),1)
            cv2.line(imgL,(int(cube_image[0,3]),int(cube_image[1,3])),(int(cube_image[0,7]),int(cube_image[1,7])),(0,0,255),1)


    labels.close()

#Definitions and classes for 3D estimation

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[0] + box_2d[2]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx.cpu()*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])
        
        
        pt1=[0,0]
        pt2=[0,0]
        
        # crop image
        pt1[0] = int(box_2d[0])
        pt1[1] = int(box_2d[1])
        pt2[0] = int(box_2d[2])
        pt2[1] = int(box_2d[3])
        
        crop = np.array([])
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
                    nn.Softmax(1)
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )


    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        #confidence = confidence.cpu().detach()

        

        dimension = self.dimension(x)
        return orientation, confidence, dimension


class ClassAverages:
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.txt'

        if len(classes) == 0: # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        class_ = class_.lower()
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        class_ = class_.lower()
        return self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        return class_.lower() in self.dimension_map