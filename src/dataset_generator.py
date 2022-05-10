import pandas as pd
pd.set_option('display.max_columns', None)
import math
import argparse
import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

from kitti_detector_system import KittiDetector3D
from pcl_img_utils import Pcl_Img_Utils

PATH_KITTI = '/media/robesafe/SSD_SATA/KITTI_DATASET/'

def main():

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--remove", help="remove the previous folder", type=bool, default=False)
    parser.add_argument("-f", "--folder", help="folder to store the new dataset", type=str, default='/media/robesafe/SSD_SATA/KITTI_FRUSTUM_DATASET/')
    parser.add_argument("-nbb", "--noise_bb", help="max noise in pixels that can be added", type=float, default=1.5)
    parser.add_argument("-nd1", "--sd_noise_distance", help="standard deviation of the noise in the distance", type=float, default=3.0)
    parser.add_argument("-nd2", "--max_noise_distance", help="max distance error in distance", type=float, default=5.0)
    args = parser.parse_args()

    # Remove the previous folder
    os.system('rm -rf '+args.folder)

    # Create all the folders
    os.mkdir(args.folder)
    os.mkdir(args.folder+'image_2/')
    os.mkdir(args.folder+'calib/')
    os.mkdir(args.folder+'velodyne/')
    os.mkdir(args.folder+'label_2/')
    os.mkdir(args.folder+'ImageSets/')

    # Copy one of the calibration files to the new dataset folder
    os.system('cp '+PATH_KITTI+'calib/000000.txt '+args.folder+'calib/')

    # Create the KittiDetector3D object
    kitti_detector = KittiDetector3D(PATH_KITTI)

    # Get the number of objects in the dataset
    #n_objects = get_n_objects()
    n_objects = 100

    # Create a Gaussian function to add noise to the distance value
    sd_noise_distance = args.sd_noise_distance
    max_noise_distance = args.max_noise_distance
    noise_distance_f = lambda x: min(x+max_noise_distance, max(x-max_noise_distance, np.random.normal(x, sd_noise_distance, n_objects)[0]))

    # Create a random noise function to add noise to the bounding box
    noise_bb = args.noise_bb
    noise_bb_f = lambda x: random.uniform(x + noise_bb, x - noise_bb)

    n_frames = 3

    id_dataset = 0
    for i in range(n_frames):
        # Load the calibration matrices, the pointcloud and the image
        P2, R0_rect, Tr_velo_to_cam = kitti_detector.get_id_calib(i)
        img, pcl = kitti_detector.get_id_img_pcl(i)

        df = load_kitti_groundtruth(i)

        # Calculate the distance between the object and the camera
        df['distance'] = (df.apply(lambda row: distance([0,0,0],[row['x'],row['y'],row['z']]), axis=1))

        # Apply error functions to the distance
        df['distance_'] = df.apply(lambda row: noise_distance_f(row['distance']), axis=1)
        # Apply error functions to the bounding box
        df['xmin'] = df.apply(lambda row: noise_bb_f(row['xmin']), axis=1)
        df['ymin'] = df.apply(lambda row: noise_bb_f(row['ymin']), axis=1)
        df['xmax'] = df.apply(lambda row: noise_bb_f(row['xmax']), axis=1)
        df['ymax'] = df.apply(lambda row: noise_bb_f(row['ymax']), axis=1)

        # Order the dataframe by distance
        df = df.sort_values(by='distance')

        # Reset the indices of the dataframe
        df.reset_index(drop=True, inplace=True)

        # Create the Pcl_Img_Utils object
        pcl_img_utils = Pcl_Img_Utils(pcl, img, P2, R0_rect, Tr_velo_to_cam, df)

        # Get the frustum pointclouds
        pcl_img_utils.calculate_point_cloud_projected()
        pcl_img_utils.calculate_projected_pcs_bb()
        frustum_pcls, _ = pcl_img_utils.get_pcs_bb()

        # Store the first id os this frame
        first_id_frame = id_dataset

        # Iterate over all the objects in the dataframe
        for index, row in df.iterrows():

            # Save the frustum pointcloud with all the values as float32 in a binary file
            pcl_file = args.folder+'velodyne/'+str(id_dataset).zfill(6)+'.bin'
            pcl_frustum = frustum_pcls[index]
            pcl_frustum = np.array(pcl_frustum).T.astype(np.float32).flatten()
            # Save the numpy array values as a binary file
            with open(pcl_file, 'wb') as f:
                f.write(pcl_frustum)

            # Save the image
            img_file = args.folder+'image_2/'+str(id_dataset).zfill(6)+'.png'
            if index == 0:
                img.save(img_file)
            else:
                # Use of symbolic links to save the images
                os.system('ln -s '+args.folder+'image_2/'+str(first_id_frame).zfill(6)+'.png '+img_file)

            # Save the calibration file from the original dataset
            calib_file = args.folder+'calib/'+str(id_dataset).zfill(6)+'.txt'
            if index == 0:
                os.system('cp '+PATH_KITTI+'calib/'+str(i).zfill(6)+'.txt '+calib_file)
            else:
                # Use of symbolic links to avoid copying the calibration files
                os.system('ln -s '+args.folder+'calib/'+str(first_id_frame).zfill(6)+'.txt '+calib_file)

            # Save the labels files

            # Save the ImageSets files

            id_dataset += 1

def distance(p1, p2):
    """
    Calculates the distance between three points in 3D.
    """
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def get_n_objects():
    """
    Returns the number of objects that are 'Car', 'Pedestrian' or 'Cyclist' in the dataset.
    """
    print("Calculating the number of objects...")
    n_frames = get_n_frames()
    # Create loading bar
    pbar = tqdm(total=n_frames)
    n_objects = 0
    for i in range(n_frames):
        df = load_kitti_groundtruth(i)
        n_objects += len((df[(df['type'] == 'Car')|
                         (df['type'] == 'Pedestrian')|
                         (df['type'] == 'Cyclist')]))
        pbar.update(1)
    pbar.close()
    print("Number of objects: ", n_objects)
    return n_objects

def get_n_frames():
    """
    Returns the number of frames in the KITTI dataset.
    """
    return len(os.listdir(PATH_KITTI+'label_2/'))

def load_kitti_groundtruth(id_frame):
    """
    Loads the groundtruth data from the groundtruth data file.
    """
    name = '%06d'%id_frame
    gt_file = PATH_KITTI+'label_2/'+name+'.txt'

    col = ['type','truncated','occluded','alpha','xmin','ymin','xmax','ymax','height','width','length','x','y','z','ry']

    df = pd.read_csv(gt_file, sep=' ', names=col)
    # Choose only the objects of interest
    df = df[(df['type'] == 'Car') | (df['type'] == 'Cyclist') | (df['type'] == 'Pedestrian')]

    return df
    
if __name__ == '__main__':
    main()