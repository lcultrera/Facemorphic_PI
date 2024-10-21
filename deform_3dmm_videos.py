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
import json

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

use_mediapipe = True

# Questo da mettere in un main ------------------------------------------------------------
# 3DMM fitting regularization
_lambda = 100
# 3DMM fitting rounds (default 1)
_rounds = 0
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
if use_mediapipe:
    idx_landmarks_3D = np.load('3D_data/mediapipe_3dmm_landmarks_idx.npy')
else:
    idx_landmarks_3D = sio.loadmat('3D_data/coma_landmarks.mat')
    idx_landmarks_3D = idx_landmarks_3D['landmarks_idx']
    idx_landmarks_3D[17] = 3709
    idx_landmarks_3D[26] = 695
    idx_landmarks_3D = idx_landmarks_3D - 1 # Python is 0-indexed
# ------------------------------------------------------------------------------------------
if use_mediapipe:
    base_options = python.BaseOptions(model_asset_path='data/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                      output_face_blendshapes=True,
                                      output_facial_transformation_matrixes=True,
                                      num_faces=1)
    fa = vision.FaceLandmarker.create_from_options(options)
else:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=True)

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
        if display_mesh:
            o3d.visualization.draw_geometries([mesh])
        else:
            # clear figure
            vis.clear_geometries()
            vis.add_geometry(mesh)
            #vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            if save_path:
                vis.capture_screen_image(save_path)
            #vis.destroy_window()
            

def get_landmarks(image_path, fa, use_mediapipe=True):
    # TODO: check if image is none so to skip it
    if use_mediapipe:
        try:
            image = mp.Image.create_from_file(image_path)
        except:
            return [], None, []
        detection_result = fa.detect(image)
        image = image.numpy_view()
        if len(detection_result.face_landmarks) == 0:
            return [], image, []
        landmarks = np.array([[landmark.x*image.shape[1], landmark.y*image.shape[0]] 
                    for landmark in detection_result.face_landmarks[0]])
        # Extract the face blendshapes category names and scores.
        face_blendshapes = detection_result.face_blendshapes[0]
        # face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        return landmarks, image, face_blendshapes_scores
    
    else:
        image = cv2.imread(image_path)
        landmarks = fa.get_landmarks(image)
        if landmarks is None or len(landmarks) == 0:
            landmarks = []
        else:
            landmarks = landmarks[0]
        return landmarks, image
    
def get_images_and_landmarks(video_path, fa, need_annot=False, use_mediapipe=True):
    # Function that takes a video path and returns a list of images and a list of landmarks
    # Images are not really necessary, but they can be used to visualize the results

    list_imgs = []
    list_landmarks = []
    list_blenshapes = []

    frame_paths = glob.glob(f'{video_path}/*.jpg')
    # natural sort
    frame_paths.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    if len(frame_paths) == 0:
        print(f'No frames found for {video_path}')
        return [], [], [], []

    # load annotation
    if need_annot:
        annot_path = f'{video_path.replace("/frames_", "/event_frames_")}/annotation.json'
        if not os.path.isfile(annot_path):
            print(f'No annotation found for {video_path}')
            return [], [], []
        with open(annot_path,'r') as af:
            cur_annot = json.load(af)
        cur_start = cur_annot['start']
        cur_end = min(cur_annot['end'], len(frame_paths))
    else:
        cur_start = 0
        cur_end = len(frame_paths)
        
    first_landmarks = None
    if os.path.exists(video_path + '/landmarks_mediapipe.npz'):
        landmarks = np.load(video_path + '/landmarks_mediapipe.npz', allow_pickle=True)['arr_0'][None][0]['landmarks']
        
        if len(landmarks) == 0 or cur_start == 99999999:
            return [], [], []

        first_frame_id = max(cur_start-1, 0)
        first_landmarks = landmarks[first_frame_id]
        list_landmarks = [landmarks[x] for x in range(cur_start, cur_end)]

        for img in tqdm(frame_paths[cur_start:cur_end]):
            if img.endswith('.jpg'):
                list_imgs.append(cv2.imread(img))
        assert len(list_landmarks) == len(list_imgs)
    else:
        for img in tqdm(frame_paths[cur_start:cur_end]):
            if img.endswith('.jpg'):
                if use_mediapipe:
                    landmarks, cur_image, face_blendshapes = get_landmarks(img, fa, use_mediapipe)
                    list_blenshapes.append(face_blendshapes)
                else:
                    landmarks, cur_image = get_landmarks(img, fa, use_mediapipe)
                    face_blendshapes = None
                    
                list_imgs.append(cur_image)
                list_landmarks.append(landmarks)
                if first_landmarks is None:
                    first_landmarks = landmarks
                
    return list_imgs, list_landmarks, first_landmarks, face_blendshapes

def fit_3dmm_vid(video_path, avgModel, avgModel_facets, idx_landmarks_3D, ID_COMPONENTS, ID_COMPONENTS_RES, ID_WEIGHTS, AU_COMPONENTS, AU_COMPONENTS_RES, AU_WEIGHTS, params3dmm, fa):
    
    # Set results
    results = {}
    alphas = []
    meshes = []
    # Get list of images and landmarks
    list_imgs, list_lndmrks, first_landmarks, face_blendshapes = get_images_and_landmarks(video_path, fa, need_annot=False, use_mediapipe=use_mediapipe)
    if list_imgs == [] or list_lndmrks == []:
        return None
    
    # Fit the identity 3DMM on the first frame
    first_index = 0
    done_first = False
    while not done_first:
        try:
            fitted_model = _3DMM_obj.opt_3DMM_fast(ID_WEIGHTS, ID_COMPONENTS, ID_COMPONENTS_RES, idx_landmarks_3D, first_landmarks, avgModel,
                            params3dmm['lambda'], params3dmm['rounds'], params3dmm['r'], params3dmm['Cdist'])
            done_first = True
        except:
            first_index += 1
            print(f'No landmarks found for the first frame, trying with frame {first_index}')
            if first_index >= len(list_lndmrks):
                print('No landmarks found for the first frame')
                return
            first_landmarks = list_lndmrks[first_index]
    
    # Store the identity model and alphas. Identity model is used to initialize the fitting on the next frames
    identity_model = fitted_model["defShape"]
    identity_alphas = fitted_model["alpha"]
    
    for im,lm in tqdm(zip(list_imgs, list_lndmrks), total=len(list_imgs)):
        # For each subsequent frame, get the detected landmarks
        detected_landmarks = lm
        if len(detected_landmarks) > 0:
            # Fit the 3DMM using the AU components and the identity model on each subsequent frame
            fitted_model = _3DMM_obj.opt_3DMM_fast(AU_WEIGHTS, AU_COMPONENTS, AU_COMPONENTS_RES, idx_landmarks_3D, detected_landmarks , identity_model,
                        params3dmm['lambda'], params3dmm['rounds'], params3dmm['r'], params3dmm['Cdist'])
            # Store the fitted model and alphas
            meshes.append(fitted_model["defShape"])
            alphas.append(fitted_model["alpha"])
        else:
            meshes.append([])
            alphas.append([])

        #deformed_face = _3DMM_obj.deform_3D_shape_fast(identity_model.swapaxes(1,0), AU_COMPONENTS, fitted_model["alpha"])
        #create_mesh(deformed_face, triangles=avgModel_facets, display_mesh=True)
        #create_mesh(fitted_model["defShape"].swapaxes(1,0), triangles=avgModel_facets, display_mesh=True)
        
    results['meshes'] = meshes
    results['alphas'] = alphas 
    results['identity_model'] = identity_model
    results['identity_alphas'] = identity_alphas
    results['landmarks'] = list_lndmrks
    results['face_blendshapes'] = face_blendshapes # CHECK IF THEY ARE IN THE SAME ORDER AS THE FRAMES
    return results
    

demo = False
if demo:
    video_path = 'recordings/00d38697d37c454c86599154f7de69da/AU_27/frames_1/'
    fit_3dmm_vid(video_path, avgModel, avgModel_facets, idx_landmarks_3D, ID_COMPONENTS, ID_COMPONENTS_RES, ID_WEIGHTS, AU_COMPONENTS, AU_COMPONENTS_RES, AU_WEIGHTS, params3dmm, fa)

else:
    rgb_folder_paths = glob.glob('recordings/*/*/frames_*/')
    #print(rgb_folder_paths)
    for rgb_folder_path in tqdm(rgb_folder_paths):
        if 'AU_READING' in rgb_folder_path or 'AU_FREE' in rgb_folder_path:
            continue

        #if os.path.exists(rgb_folder_path + '/landmarks.npz') and os.path.exists(rgb_folder_path + '/alphas32.npz'):
        if os.path.exists(rgb_folder_path + '/alphas32_mediapipe.npz'):
            print(f'Skipping {rgb_folder_path}')
            continue

        res = fit_3dmm_vid(rgb_folder_path, avgModel, avgModel_facets, idx_landmarks_3D, ID_COMPONENTS, ID_COMPONENTS_RES, ID_WEIGHTS, AU_COMPONENTS, AU_COMPONENTS_RES, AU_WEIGHTS, params3dmm, fa)
        # Save results
        if res:
            np.savez_compressed(rgb_folder_path + '/landmarks_mediapipe.npz', {'landmarks': res['landmarks']})
            np.savez_compressed(rgb_folder_path + '/alphas32_mediapipe.npz', {'alphas': res['alphas'], 'identity_alphas': res['identity_alphas']})
            print(f'Saved {rgb_folder_path}')
        else:
            print(f'No landmarks found for {rgb_folder_path}')