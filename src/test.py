from kitti_detector_system import KittiDetector3D
from detecction_system import Detector3D
from distance_approx_functions import *

PATH_KITTI = '/media/robesafe/SSD_SATA/KITTI_DATASET/'
n_frames = 2

# Load the KITTI detector and the detection system
kitti_detector = KittiDetector3D(PATH_KITTI)
detector = Detector3D()

# Load the calibration matrices and the models
detector.set_calibration_matrices(kitti_detector.get_id_calib(0))
detector.load_models()

# Get 3D detections
import time
for i in range(n_frames):
    start = time.time()
    detector.set_image_pointcloud(kitti_detector.get_id_img_pcl(i))
    detector.detect_3d()
    print(time.time() - start)
    print(detector.distance_approx.detections)
