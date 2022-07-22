import pandas as pd
pd.set_option('display.max_columns', None)
import math
import argparse
import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import random
import subprocess

from kitti_detector_system import KittiDetector3D
from pcl_img_utils import Pcl_Img_Utils

def main():

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-pf", "--previous_folder", help="folder where KITTI dataset is stored", type=str, default='/media/robesafe/SSD_SATA/KITTI_DATASET/')
    parser.add_argument("-f", "--folder", help="folder to store the new dataset", type=str, default='/media/robesafe/SSD_SATA/KITTI_FRUSTUM_DATASET/')
    parser.add_argument("-nbb", "--noise_bb", help="max noise in pixels that can be added", type=float, default=2)
    parser.add_argument("-nd", "--noise_distance", help="max noise in the distance that can be added", type=float, default=5)
    parser.add_argument("-r", "--repeat", help="repeat n times the dataset", type=int, default=1)
    parser.add_argument("-n", "--normalize", help="normalization of class distribution", type=str, default=False)
    parser.add_argument("-md", "--max_distance", help="maximum distance to filtrate the pointclouds", type=float, default=5.5)
    parser.add_argument("-o", "--offset", help="offset in z coordinate (from camera)", type=float, default=0)
    args = parser.parse_args()
    if args.normalize == "True":
        args.normalize = True
    elif args.normalize == "False" or args.normalize == False:
        args.normalize = False
    else:
        print('Input a boolean value to -n/--normalize flag')
        exit(1)
        

    # Remove the previous folder
    os.system('rm -rf '+args.folder)

    # Create all the folders
    os.mkdir(args.folder)
    os.mkdir(args.folder+'image_2/')
    os.mkdir(args.folder+'calib/')
    os.mkdir(args.folder+'velodyne/')
    os.mkdir(args.folder+'label_2/')
    os.mkdir(args.folder+'ImageSets/')
    subprocess.run(['touch',args.folder+'ImageSets/train.txt'])
    subprocess.run(['touch',args.folder+'ImageSets/val.txt'])

    # Copy one of the calibration files to the new dataset folder
    os.system('cp '+args.previous_folder+'calib/000000.txt '+args.folder+'calib/')

    # Create the KittiDetector3D object
    kitti_detector = KittiDetector3D(args.previous_folder)

    # Create a Gaussian function to add noise to the distance value
    noise_distance = args.noise_distance
    noise_distance_f = lambda x: random.uniform(x + noise_distance, x - noise_distance)

    # Create a random noise function to add noise to the bounding box
    noise_bb = args.noise_bb
    noise_bb_f = lambda x: random.uniform(x + noise_bb, x - noise_bb)

    # Calculate the number of objects per type
    n_objects = 1
    n_car, n_pedestrian, n_cyclist = 1, 1, 1
    if args.normalize:
        n_objects, (n_car, n_pedestrian, n_cyclist) = get_n_objects(args.previous_folder)

    # Calculate the probability to get other object of that type
    p_car = n_car/n_objects
    p_pedestrian = n_pedestrian/n_objects
    p_cyclist = n_cyclist/n_objects
    max_p_type = max(p_car, p_pedestrian, p_cyclist)
    p_dict = {'Car':p_car, 'Pedestrian':p_pedestrian, 'Cyclist':p_cyclist}

    # Calculate number of frames in the dataset
    n_frames = get_n_frames(args.previous_folder)
    #n_frames=100

    # Create loading bar
    pbar = tqdm(total=n_frames*args.repeat)

    # Create list to track the time necesary to modify the pcl
    time_list = []

    id_dataset = 0
    for i in range(n_frames):
        # Load the calibration matrices, the pointcloud and the image
        P2, R0_rect, Tr_velo_to_cam = kitti_detector.get_id_calib(i)
        img, pcl = kitti_detector.get_id_img_pcl(i)

        df = load_kitti_groundtruth(i, args.previous_folder)

        # Calculate the distance between the object and the camera
        df['distance_ensemble'] = (df.apply(lambda row: distance([0,0,0],[row['x'],row['y'],row['z']]), axis=1))
        
        # Apply error functions to the bounding box
        df['xmin'] = df.apply(lambda row: noise_bb_f(row['xmin']), axis=1)
        df['ymin'] = df.apply(lambda row: noise_bb_f(row['ymin']), axis=1)
        df['xmax'] = df.apply(lambda row: noise_bb_f(row['xmax']), axis=1)
        df['ymax'] = df.apply(lambda row: noise_bb_f(row['ymax']), axis=1)

        # Order the dataframe by distance
        df = df.sort_values(by='distance_ensemble')

        # Reset the indices of the dataframe
        df.reset_index(drop=True, inplace=True)

        # Create the Pcl_Img_Utils object
        pcl_img_utils = Pcl_Img_Utils(pcl, img, P2, R0_rect, Tr_velo_to_cam, df)

        # Get the frustum pointclouds
        pcl_img_utils.calculate_point_cloud_projected()
        pcl_img_utils.calculate_projected_pcs_bb()
        frustum_pcls, points3d_det, angles, compute_time = pcl_img_utils.get_gt_frustum(noise_distance_f, args.max_distance, args.offset)

        # Save the compute_time in the time_list
        time_list.append(compute_time)

        # Convert df[['x', 'y', 'z']] to a numpy array of shape (n_objects, 3)
        points3d = np.array(df[['x', 'y', 'z']]).reshape((-1,3))
        # Add column to points3d with all 1's
        points3d = np.hstack((points3d, np.ones((points3d.shape[0], 1))))
        # Transform to LiDAR coordinate system
        points3d = list(map(lambda point3d: np.array(point3d).reshape(4,1), points3d))

        # Rotate points3d over z
        rots_z = list(map(lambda angle: np.array([[np.cos(angle),0,np.sin(angle),0],
                                                  [0,1,0,0],
                                                  [-np.sin(angle),0,np.cos(angle),0],
                                                  [0,0,0,1]], dtype=np.float), angles))
        points3d = np.array(list(map(lambda point3d, rot_z: np.dot(rot_z,point3d), points3d, rots_z)))

        # Reshape the points3d_det
        points3d_det = np.array(points3d_det).reshape(-1,4).T
        points3d_det[3,:] = 1
        points3d_det = np.reshape(np.array(points3d_det).T,(points3d_det.shape[1],4,1))

        # Move the pointcloud
        points3d = np.array(list(map(lambda point3d, point_3d: point3d - point_3d, points3d, points3d_det)))
        
        # Store in the dataframe
        df['x'] = points3d[:,0].flatten().tolist()
        df['y'] = points3d[:,1].flatten().tolist()
        df['z'] = points3d[:,2].flatten().tolist()

        # Add angles to each value of 'ry' in the dataframe
        df['ry'] = list(map(lambda ry, angle: ry+float(angle.flatten()), df['ry'].tolist(), angles))

        # Store the first id os this frame
        first_id_frame = id_dataset

        # Check if the frame is part of the training or validation set (by looking into the ImageSets folder)
        training = None
        if os.path.exists(args.previous_folder+'ImageSets/train.txt'):
            with open(args.previous_folder+'ImageSets/train.txt', 'r') as f:
                lines = f.readlines()
                lines = list(map(lambda x: x.split('/')[-1][:-1], lines))
                if str(i).zfill(6)+'.png' in lines:
                    training = True
                else:
                    training = False

        first_img_calib = True
        # Iterate over all the objects in the dataframe
        for index, row in df.iterrows():

            # Check if the bounding box has points inside it
            frustum_pcl_index = np.dot(np.dot(R0_rect, Tr_velo_to_cam),frustum_pcls[index].copy())
            frustum_pcl_index = np.delete(frustum_pcl_index,np.where((frustum_pcl_index[1,:]<float(row['x']-row['length']))|\
                                                                     (frustum_pcl_index[1,:]>float(row['x']+row['length']))|\
                                                                     (frustum_pcl_index[0,:]<float(row['z']-row['length']))|\
                                                                     (frustum_pcl_index[0,:]>float(row['z']+row['length']))),axis=1)
            if frustum_pcl_index.shape[1] <= 3:
                continue

            # Check times to calculate each object
            times = max_p_type/p_dict[row['type']]
            times_ = random.uniform(0,1)<=times%1
            times = times//1+times_
            for repetition in range(int(times*args.repeat)):

                # Save the frustum pointcloud with all the values as float32 in a binary file
                pcl_file = args.folder+'velodyne/'+str(id_dataset).zfill(6)+'.bin'
                pcl_frustum = frustum_pcls[index]
                pcl_frustum = np.array(pcl_frustum).T.astype(np.float32).flatten()
                # Save the numpy array values as a binary file
                with open(pcl_file, 'wb') as f:
                    f.write(pcl_frustum)

                # Save the image
                img_file = args.folder+'image_2/'+str(id_dataset).zfill(6)+'.png'
                if first_img_calib and repetition == 0:
                    img.save(img_file)
                else:
                    # Use of symbolic links to save the images
                    past_img_path = args.folder+'image_2/'+str(first_id_frame).zfill(6)+'.png'
                    os.symlink(past_img_path, img_file)
                    
                # Save the calibration file from the original dataset
                calib_file = args.folder+'calib/'+str(id_dataset).zfill(6)+'.txt'
                if first_img_calib and repetition == 0:
                    os.system('cp '+args.previous_folder+'calib/'+str(i).zfill(6)+'.txt '+calib_file)
                else:
                    # Use of symbolic links to avoid copying the calibration files
                    past_calib_path = args.folder+'calib/'+str(first_id_frame).zfill(6)+'.txt'
                    os.symlink(past_calib_path, calib_file)

                    
                # Save the label file with KITTI format
                label_file = args.folder+'label_2/'+str(id_dataset).zfill(6)+'.txt'
                with open(label_file, 'w') as f:
                    for elem in list(row[:15]):
                        # Write the values in the file with an extra space between them if it is not the last one
                        if elem != row['ry']:
                            if isinstance(elem, float):
                                elem = '{:.2f}'.format(elem)
                            f.write(str(elem)+' ')
                        else:
                            if isinstance(elem, float):
                                elem = '{:.2f}'.format(elem)
                            f.write(str(elem))
                    f.write('\n')

                # Write on the correspoding ImageSet file
                if training:
                    with open(args.folder+'ImageSets/train.txt', 'a') as f:
                        f.write(img_file+'\n')
                else:
                    with open(args.folder+'ImageSets/val.txt', 'a') as f:
                        f.write(img_file+'\n')

                id_dataset += 1
                first_img_calib = False

        pbar.update(1)
    
    # Describe the metrics of the time_list
    print('\n')
    print('Mean time: '+str(np.mean(time_list)))
    print('Median time: '+str(np.median(time_list)))
    print('Max time: '+str(np.max(time_list)))
    print('Min time: '+str(np.min(time_list)))
    print('Std time: '+str(np.std(time_list)))
    print('\n')

    # Create simbolic link to a training folder
    os.mkdir(args.folder+'training/')
    os.system('cd '+args.folder+'training/; ln -s ../* .')

def distance(p1, p2):
    """
    Calculates the distance between three points in 3D.
    """
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def get_n_frames(previous_folder):
    """
    Returns the number of frames in the KITTI dataset.
    """
    return len(os.listdir(previous_folder+'label_2/'))

def load_kitti_groundtruth(id_frame, previous_folder):
    """
    Loads the groundtruth data from the groundtruth data file.
    """
    name = '%06d'%id_frame
    gt_file = previous_folder+'label_2/'+name+'.txt'

    col = ['type','truncated','occluded','alpha','xmin','ymin','xmax','ymax','height','width','length','x','y','z','ry']

    df = pd.read_csv(gt_file, sep=' ', names=col)
    # Choose only the objects of interest
    df = df[(df['type'] == 'Car') | (df['type'] == 'Cyclist') | (df['type'] == 'Pedestrian')]

    return df

def get_n_objects(previous_folder):
    """
    Returns the number of objects that are 'Car', 'Pedestrian' or 'Cyclist' in the dataset.
    """
    print("Calculating the number of objects...")
    n_frames = get_n_frames(previous_folder)
    # Create loading bar
    pbar = tqdm(total=n_frames)
    n_objects = 0
    n_car, n_pedestrian, n_cyclist = 0, 0, 0
    for i in range(n_frames):
        df = load_kitti_groundtruth(i, previous_folder)
        n_objects += len((df[(df['type'] == 'Car')|
                         (df['type'] == 'Pedestrian')|
                         (df['type'] == 'Cyclist')]))
        n_car += len(df[df['type'] == 'Car'])
        n_pedestrian += len(df[df['type'] == 'Pedestrian'])
        n_cyclist += len(df[df['type'] == 'Cyclist'])
        pbar.update(1)
    pbar.close()
    print("Number of objects: ", n_objects)
    n_types = (n_car, n_pedestrian, n_cyclist)
    return n_objects, n_types
    
if __name__ == '__main__':
    main()