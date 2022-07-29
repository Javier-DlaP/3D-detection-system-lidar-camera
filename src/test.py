from kitti_detector_system import KittiDetector3D
from detecction_system import Detector3D
from distance_approx_functions import *
from tqdm import tqdm
import warnings
import os
import shutil

PATH_KITTI = '/media/robesafe/SSD_SATA/KITTI_DATASET/'
#n_frames = 100
n_frames = len(os.listdir(PATH_KITTI+'velodyne/'))

# Load the KITTI detector and the detection system
kitti_detector = KittiDetector3D(PATH_KITTI)
detector = Detector3D()
detector.load_models()

# Remove pred folder
if os.path.isdir(kitti_detector.path_kitti_pred):
    shutil.rmtree(kitti_detector.path_kitti_pred)
os.mkdir(kitti_detector.path_kitti_pred)

# Remove runtime warnings (from point2d -> point3d calculation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get 3D detections
for i in tqdm(range(n_frames)):
    detector.set_calibration_matrices(kitti_detector.get_id_calib(i))
    detector.set_image_pointcloud(kitti_detector.get_id_img_pcl(i))

    detector.detect_3d()

    kitti_detector.write_prediction(detector.final_detections, i)

# Visualize execution times
detector.show_times()