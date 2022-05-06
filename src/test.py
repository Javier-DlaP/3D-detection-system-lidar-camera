from kitti_detector_system import KittiDetector3D
from detecction_system import Detector3D
from distance_approx_functions import *

PATH_KITTI = '/media/robesafe/SSD_SATA/KITTI_DATASET/'
id_frame = 1

# Load the KITTI detector and the detection system
kitti_detector = KittiDetector3D(PATH_KITTI)
detector = Detector3D()

# Load the calibration matrices, image and pointcloud
detector.set_calibration_matrices(kitti_detector.get_id_calib(0))
detector.set_image_pointcloud(kitti_detector.get_id_img_pcl(id_frame))
detector.load_models()

# Get 3D detections
import time
start = time.time()
detector.detect_3d()
print(time.time() - start)

print(detector.distance_approx.detections)
