"""
Utils for visual effects
"""
import numpy as np
import open3d

def pointcloud(depth, fov_x, fov_y):
    fx = 0.5 / np.tan(fov_x * 0.5) 
    fy = 0.5 / np.tan(fov_y * 0.5)
    height = depth.shape[0]
    width = depth.shape[1]
    mask = np.where(depth > 0)  
    x = mask[1]
    y = mask[0]
    normalized_x = (x.astype(np.float32) - width * 0.5) / width
    normalized_y = (y.astype(np.float32) - height * 0.5) / height
    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]
    ones = np.ones(world_z.shape[0], dtype=np.float32)
    return np.vstack((world_x, world_y, world_z, ones)).T

def normalize_depth(prediction):
    prediction = prediction - prediction.min()
    prediction = prediction / prediction.max()
    return 1 - prediction

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    return inlier_cloud

