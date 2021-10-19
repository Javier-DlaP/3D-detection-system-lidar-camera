# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import seaborn as sn
import sklearn as skl

# Constants
KITTI_DATASET = '/media/javier/HDD_linux/KITTI_dataset/'

# Open CSV file and transform to pandas dataframe
df = pd.read_csv(KITTI_DATASET+'training/kitti_gt.csv')
print(df)

# Plot histogram of a column and name the x axis
df['lateral_distance'].hist(bins=100)
plt.xlabel('lateral_distance')

# Show all statistical analysis of a column
print(df['lateral_distance'].describe())

# Show a subplot containing 4 histograms of different columns
df['lateral_distance'].hist(bins=100)
plt.xlabel('lateral_distance')

# Convert column to 1D array
array = df['lateral_distance'].values



X = df['distance'].values.reshape(-1, 1)
Y = df['heigth_image']

reg = skl.pipeline.make_pipeline(skl.preprocessing.StandardScaler(),
                                 skl.linear_model.SGDRegressor(max_iter=1000, tol=0))
reg.fit(X,Y)

# Plot the regression curve
y_pred = reg.predict(X)
plt.scatter(X, Y)
plt.plot(X, y_pred, color='red')
plt.xlabel('distance')
plt.ylabel('heigth_image')


# plt.scatter(X, Y)
# plt.plot(X, Y_pred, color='red')
# plt.title('Stochastic Gradient Descent')
# plt.show()
