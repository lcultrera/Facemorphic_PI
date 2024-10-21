import os
import cv2
from _3DMM import _3DMM
from Matrix_operations import Matrix_op
import h5py
import scipy.io as sio
import face_alignment
from tqdm import tqdm
import open3d as o3d
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Questo da mettere in un main ------------------------------------------------------------
# 3DMM fitting regularization
_lambda = 1
# 3DMM fitting rounds (default 1)
_rounds = 1
# Parameters for the HPR operator (do not change)
_r = 3
_C_dist = 700
# Params dict
params3dmm = {'lambda': _lambda, 'rounds': _rounds, 'r': _r, 'Cdist': _C_dist}
# Instantiate all objects
_3DMM_obj = _3DMM()
# Load all 3DMM data
# Template
template = sio.loadmat('3D_data/template_D3DFACS.mat')
avgModel = template['template_n']
avgModel_facets = template['f']
avgModel_facets = avgModel_facets - 1 # Python is 0-indexed
# Identity components
ID = sio.loadmat('3D_data/PCA_D3DFACS_ID_22.mat')
ID_COMPONENTS = ID['Components']
ID_WEIGHTS = ID['Weights']
# AU components
AU = sio.loadmat('3D_data/DL_D3DFACS_expr_0.05_256.mat') # oppure PCA_D3DFACS_expr_256 ma DL funziona meglio mi pare
AU = sio.loadmat('3D_data/DL_D3DFACS_expr_0.05_32.mat')
AU_COMPONENTS = AU['Components_e']
AU_WEIGHTS = AU['Weights_e']
# Setup 3D objects and reshape components
# Identity model
m_X_obj = Matrix_op(ID_COMPONENTS, None)
m_X_obj.reshape(ID_COMPONENTS)
ID_COMPONENTS_RES = m_X_obj.X_res
# AU model
m_X_obj = Matrix_op(AU_COMPONENTS, None)
m_X_obj.reshape(AU_COMPONENTS)
AU_COMPONENTS_RES = m_X_obj.X_res
# Landmarks indices
idx_landmarks_3D = sio.loadmat('3D_data/coma_landmarks.mat')
idx_landmarks_3D = idx_landmarks_3D['landmarks_idx']
idx_landmarks_3D[17] = 3709
idx_landmarks_3D[26] = 695
idx_landmarks_3D = idx_landmarks_3D - 1 # Python is 0-indexed
# ------------------------------------------------------------------------------------------


#vis = o3d.visualization.VisualizerWithKeyCallback()

def vis_PC(PC):
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.add_geometry(PC)
  ctr = vis.get_view_control()  # Everything good
  vis.run()

            
def create_mesh(deformed_face, triangles, display_mesh=False, save_path=None, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(deformed_face.swapaxes(1,0))

        pcd.estimate_normals()
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        #radius = 1.5 * avg_dist   

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector())

        mesh.triangles = o3d.utility.Vector3iVector(triangles.swapaxes(1,0))
        # o3d.visualization.draw_geometries([mesh])
        # print("A mesh with no normals and no colors does not seem good.")
        print("Computing normal and rendering it.")
        mesh.compute_vertex_normals()
        #print(np.asarray(mesh.triangle_normals))
        return mesh

#video_path = 'recordings/a774e2bd46dd41f5b0e48c3a4d1277b4/AU_27/frames_2'
#video_path = 'recordings/9cd270403b024c34b651f9fe20feff2f/AU_2/frames_2'
video_path = 'recordings/9cd270403b024c34b651f9fe20feff2f/AU_55/frames_2'

loaded_alpha = np.load(f'{video_path}/alphas32_mediapipe.npz', allow_pickle=True)['arr_0'][None][0]
alphas = loaded_alpha['alphas']
identity_alphas = loaded_alpha['identity_alphas'][:,None]
vis = o3d.visualization.Visualizer()
vis.create_window()

landmarks = np.load(f'{video_path}/landmarks.npz', allow_pickle=True)['arr_0'][None][0]['landmarks']
landmarks = np.array(landmarks)
t = np.array([np.ones(100)*i for i in range(len(landmarks))]).flatten()

'''def update_graph(num):
    graph._offsets3d = landmarks[num,:,0], landmarks[0,:, 2], landmarks[0,:,1]
    title.set_text('3D Test, time={}'.format(num))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

graph = ax.scatter(landmarks[0,:,0], landmarks[0,:,1], landmarks[0,:,2])

ani = FuncAnimation(fig, update_graph, len(landmarks), 
                               interval=40, blit=False)

plt.show()'''



# read frames
frames = glob.glob(f"{video_path}/*.jpg")

# natural argsort
sorted_ids = np.argsort([int(f.split('/')[-1].split('_')[1]) for f in frames])

frames = [frames[x] for x in sorted_ids]
#landmarks = [landmarks[x] for x in sorted_ids]
#np.savez_compressed(lan_path, {'landmarks': landmarks})

# save video
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('landmarks.avi', fourcc, 30, (640, 480))

# read frames
# for n, f in enumerate(frames):
#     img = cv2.imread(f)
#     for l in landmarks[n]:
#         cv2.circle(img, (int(l[0]), int(l[1])), 3, (0, 0, 255), -1)
#     #out.write(img)
#     cv2.imshow('frame', img)
#     cv2.waitKey(100)

#out.release()

vis = o3d.visualization.Visualizer()
vis.create_window()
for n, a in enumerate(alphas):
    deformed_face = _3DMM_obj.deform_3D_shape_fast(avgModel.swapaxes(1,0), AU_COMPONENTS, a)
    mesh = create_mesh(deformed_face, triangles=avgModel_facets)
    vis.clear_geometries()
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)


    


