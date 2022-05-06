import numpy as np

def objective_0(x):
        return 0

def objective_const(x, a):
        return a

def objective_log_f(x, a, b, c):
    return a * np.log(x)**b + c

def objective_f(x, a, b):
    return a * x + b

def objective_2f(x, a, b, c):
    return a * x**2 + b * x + c

def objective_3f(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def objective_4f(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def objective_5f(x, a, b, c, d, e, f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

def objective_6f(x, a, b, c, d, e, f, g):
    return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

def objective_7f(x, a, b, c, d, e, f, g, h):
    return a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h

def objective_8f(x, a, b, c, d, e, f, g, h, i):
    return a * x**8 + b * x**7 + c * x**6 + d * x**5 + e * x**4 + f * x**3 + g * x**2 + h * x + i

def objective_9f(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**9 + b * x**8 + c * x**7 + d * x**6 + e * x**5 + f * x**4 + g * x**3 + h * x**2 + i * x + j

class Regression_model:
    def __init__(self, function, variables):
        self.function = function
        self.variables = variables
    def __str__(self):
        return "function = %s, variables = %s"%(self.function, self.variables)

class Regression_model_class:
    def __init__(self, model_car, model_pedestrian, model_cyclist):
        self.car = model_car
        self.pedestrian = model_pedestrian
        self.cyclist = model_cyclist
    def __str__(self):
        return "car: {%s},\npedestrian: {%s},\ncyclist: {%s}"%(self.car, self.pedestrian, self.cyclist)
        
class Regression_model_bb:
    def __init__(self, bb_complete, bb_incomplete):
        self.bb_complete = bb_complete
        self.bb_incomplete = bb_incomplete
    def __str__(self):
        return "bb_complete: {%s},\nbb_incomplete: {%s}"%(self.bb_complete, self.bb_incomplete)
    
class Regression_model_error:
    def __init__(self, height_distance, height_2d, pointcloud_distance, pointcloud_n_points):
        self.height_distance = height_distance
        self.height_2d = height_2d
        self.pointcloud_distance = pointcloud_distance
        self.pointcloud_n_points = pointcloud_n_points
    def __str__(self):
        return "height_distance: {%s},\nheight_2d: {%s},\npointcloud_distance: {%s},\npointcloud_n_points: {%s}"%\
                                (self.height_distance, self.height_2d, self.pointcloud_distance, self.pointcloud_n_points)