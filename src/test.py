from kitti_detector_system import KittiDetector3D
from detecction_system import Detector3D
from distance_approx_functions import *
from tqdm import tqdm
import warnings

PATH_KITTI = '/media/robesafe/SSD_SATA/KITTI_DATASET/'
n_frames = 100

# Load the KITTI detector and the detection system
kitti_detector = KittiDetector3D(PATH_KITTI)
detector = Detector3D()
detector.load_models()

# Remove runtime warnings (from point2d -> point3d calculation)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get 3D detections
for i in tqdm(range(n_frames)):
    detector.set_calibration_matrices(kitti_detector.get_id_calib(i))
    detector.set_image_pointcloud(kitti_detector.get_id_img_pcl(i))

    detector.detect_3d()
    #print(detector.detections['name'])

# Visualize execution times
detector.show_times()