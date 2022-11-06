import os
import sys
import glob
import torch
import numpy as np
from visual_utils import *
from third_party.MiDaS.utils import *
from predict_depth import run
import cv2
import open3d
from tqdm import tqdm
import argparse
import imageio
import matplotlib.pyplot as plt
from PIL import Image 
import open3d.visualization.rendering as rendering

DEFAULT_INPUT_PATH = "input/test_img.PNG"
DEFAULT_OUTPUT_PATH = "output/animation.gif"
DEFAULT_MODEL_PATH = "third_party/MiDaS/weights/dpt_large-midas-2f21e586.pt"

def cl_parser():
    parser = argparse.ArgumentParser(description="Animation properties")
    parser.add_argument('--gif_duration', default=8, type=float, help='total duration of output gif (in seconds)')
    parser.add_argument('--gif_frame_rate', default=15, type=int, help='frame rate of output gif (in frames per second)')
    parser.add_argument('--resolution', default=[720, 1280], type=list, help='target resolution of output gif')
    parser.add_argument('--image_path', default=DEFAULT_INPUT_PATH, type=str, help='path to input image')
    parser.add_argument('--output_path', default=DEFAULT_OUTPUT_PATH, type=str, help='save directory of output')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str, help='path to model weights')
    parser.add_argument('--save_pointcloud', default=False, type=bool, help='save flag for generated point cloud')
    arguments = parser.parse_args()
    return arguments

def main():
    args = cl_parser()
    resolution = args.resolution
    img = read_image(args.image_path)
    img_resized = cv2.resize(img, resolution, interpolation = cv2.INTER_AREA)
    prediction = run(img=img_resized, model_path=args.model_path,  model_type="dpt_large", optimize=True)
    fov_x = np.deg2rad(50)
    fov_y = (img_resized.shape[0]/img_resized.shape[1]) * np.deg2rad(50)
    pc = pointcloud(2*normalize_depth(prediction), fov_x = fov_x, fov_y = fov_y)
    colors = np.vstack((img_resized[:,:,0].flatten(), 
                        img_resized[:,:,1].flatten(), 
                        img_resized[:,:,2].flatten()))
    colors = (colors.T)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc[:,:3])
    pcd.colors = open3d.utility.Vector3dVector(colors[:-1,:])
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    #inlier_points, inlier_indices = display_inlier_outlier(pcd, ind)
    cleaned_pcd = pcd.select_by_index(ind)
    assert pcd.has_colors(), "Colors could not be added to the point cloud"
    #open3d.visualization.draw_geometries([pcd])
    
    
    pinhole = open3d.camera.PinholeCameraIntrinsic(resolution[0], resolution[1], 1000, 1000, resolution[0]/2, resolution[1]/2)
    mtl = open3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [0.75, 0.75, 0.75, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"
    current_center = np.array([0, 0, 1])  # look_at target
    current_eye = np.array([0, 0, -0.5])  # camera position
    current_up = np.array([0, -180, 10])  # camera orientation+
    current_FOV = 65
    
    final_center = np.array([0.0, 0.0, 1.0])  # look_at target
    final_eye = np.array([0, 0, 0])  # camera position
    final_up = np.array([0, -180, 0])  # camera orientation
    final_FOV = 105
    delta_center = final_center - current_center
    delta_eye = final_eye - current_eye
    delta_up = final_up - current_up
    delta_FOV = final_FOV - current_FOV
    num_steps = args.gif_duration * args.gif_frame_rate
    render = rendering.OffscreenRenderer(resolution[0], resolution[1])
    # setup camera intrinsic values
    render.scene.add_geometry("generated_pointcloud", cleaned_pcd, mtl)

    if os.path.exists(args.output_path):
        os.remove(args.output_path)
    with imageio.get_writer(args.output_path, mode='I', fps = args.gif_frame_rate) as writer:
        for i in tqdm(range(num_steps)): 
            vertical_field_of_view = current_FOV  
            aspect_ratio = resolution[0]/resolution[1] 
            near_plane = 0.5
            far_plane = 2
            fov_type = open3d.visualization.rendering.Camera.FovType.Vertical
            render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)
            render.scene.camera.look_at(current_center, current_eye, current_up)
            img_o3d = np.array(render.render_to_image())
            writer.append_data(img_o3d)
            if i<=num_steps/2:
                current_center = current_center + (delta_center/num_steps)
                current_eye = current_eye + (delta_eye/num_steps)
                current_up = current_up + (delta_up/num_steps)
                current_FOV = current_FOV + (delta_FOV/num_steps)
            else:
                current_center = current_center - (delta_center/num_steps)
                current_eye = current_eye - (delta_eye/num_steps)
                current_up = current_up - (delta_up/num_steps)
                current_FOV = current_FOV - (delta_FOV/num_steps)
                
    writer.close() 

if __name__ == "__main__":
    main()