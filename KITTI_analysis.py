# Import libraries
import pandas as pd
import os
import math

KITTI_DATASET = '/media/javier/HDD_linux/KITTI_dataset'

# Calculate distance between two 3D points
def distance(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

# Get KITTIs TXT file and save it as a CSV file
def kitti_2_csv(label_path, save_path='./'):

    files = sorted(os.listdir(label_path))
    col = ['frame','id','type','truncated','occluded','alpha','left','top','right','bottom','height','width','length','x','y','z','ry']
    id = 0
    lines = []

    for f in files:

        frame = 0
        labels_f = open(label_path+f,'r')

        # Add every line of the TXT file to the pandas dataframe
        for line in labels_f:
            line = line.split()
            line = [str(line[0])]+[float(i) for i in line[1:]]
            line.insert(0,frame)
            line.insert(0,id)
            if(line[2] == 'Car' or line[2] == 'Cyclist' or line[2] == 'Pedestrian'):
                lines.append(line)

        id += 1
        labels_f.close()

    df = pd.DataFrame(lines, columns=col)
    
    df['distance'] = (df.apply(lambda row: distance([0,0,0],[row['x'],row['y'],row['z']]), axis=1))

    print(df)
    
    df.to_csv(save_path+'kitti_gt.csv',columns=col,index=False)

if __name__ == "__main__":
    kitti_2_csv(KITTI_DATASET+'/training/label_2/', KITTI_DATASET+'/training/')