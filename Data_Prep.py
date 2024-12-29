#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

import os
import sys
import tqdm
import math
import torch
import pickle
import random
import trimesh
import argparse
import pymeshfix

from scipy.optimize import curve_fit


# In[3]:


np.random.seed(42)
random.seed(42)


# In[4]:


def line_dists(points, start, end):
    if np.all(start == end):
        return np.linalg.norm(points - start, axis=1)

    vec = end - start
    cross = np.linalg.norm(np.cross(vec, start - points), axis=1)
    return np.divide(cross, np.linalg.norm(vec))


def rdp_test(M, epsilon=0):
    M = np.array(M)
    start, end = M[0], M[-1]
    dists = line_dists(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax > epsilon:
        result1 = rdp_test(M[:index + 1], epsilon)
        result2 = rdp_test(M[index:], epsilon)

        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([start, end])

    return result


# In[5]:


degree = 5; num = 32


# # Read Mesh

# In[6]:


def read_mesh(filename):
    scene_or_mesh = trimesh.load(filename)
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None
        else:
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


# # Get Pieces

# In[7]:


def interpolate(vertices, req_num, option=1):
    per_pair = math.ceil(req_num / vertices.shape[0])
    new_vertices = []
    
    for i in range(len(vertices)-1):
        v1, v2 = vertices[i], vertices[i+1]
        if option == 1:
            points = list(np.linspace(v1, v2, num=per_pair+1, endpoint=False))[1:]
        else:
            lamdas = np.linspace(0, 1, num=per_pair+1, endpoint=False)[1:]
            points = lamdas.reshape((per_pair,1)) * v1.reshape((3,1)).T + (1 - lamdas).reshape((per_pair,1)) * v2.reshape((3,1)).T
            points = list(points)
        new_vertices.append(v1)
        new_vertices = new_vertices + points
        new_vertices.append(v2)
        
    return np.array(new_vertices)


# In[8]:


def filterEndpoints(endpoints, angles_dict, total_vert, diff_num=degree+1):
    i = 0; endpoints = list(endpoints)
    while i != len(endpoints):
        if np.abs(endpoints[i] - endpoints[i-1]) < diff_num or np.abs(endpoints[i] - total_vert + endpoints[i-1]) < diff_num:
            if angles_dict[endpoints[i]] > angles_dict[endpoints[i-1]]:
                endpoints.pop(i-1)
            else:
                endpoints.pop(i)
        else:
            i += 1
    return np.asarray(endpoints)


# In[9]:


def getAngles(all_vertices, endpoints, diff_angle=2):
    angles_dict = {}
    for index in range(len(endpoints)):
        i = int(endpoints[index])
        if i == 0:
            p0 = all_vertices[i-2-diff_angle]; p1 = all_vertices[i]; p2 = all_vertices[(i+1+diff_angle)%all_vertices.shape[0]]
        elif i == len(all_vertices) - 1:
            p0 = all_vertices[i-1-diff_angle]; p1 = all_vertices[i]; p2 = all_vertices[(i+1+diff_angle)%all_vertices.shape[0]+1]
        else:
            p0 = all_vertices[i-1-diff_angle]; p1 = all_vertices[i]; p2 = all_vertices[(i+1+diff_angle)%all_vertices.shape[0]]
        
        angle = np.arccos(np.abs(np.dot(p0-p1, p2-p1)) / (np.linalg.norm(p0-p1) * np.linalg.norm(p2-p1))) * 180 / np.pi
        angles_dict[i] = angle
        
    angles_dict = {k: v for k, v in sorted(angles_dict.items(), key=lambda item: -item[1])}
    
    return angles_dict


# In[10]:


def getPiecewiseCurvesAdaptive(all_vertices, num=16):
    if all_vertices.shape[0] <= num * (degree + 2):
        all_vertices = interpolate(all_vertices, num * (degree + 2))
    min_bounds = np.min(all_vertices, axis=0); max_bounds = np.max(all_vertices, axis=0)
    diagonal_len = np.linalg.norm(max_bounds - min_bounds)
    
    if num == 1:
        return [all_vertices]
    
    endpoints = np.asarray([]); eps = diagonal_len; c = 1; vertices = np.copy(all_vertices[:-1])
    while len(endpoints) < num * 2:
        mask = rdp_test(vertices, epsilon=eps/c)
        mask = np.nonzero(np.isin(vertices, mask))[0]
        mask, count = np.unique(mask, return_counts=True)
        mask = mask[count==3]
        endpoints = np.concatenate((endpoints, mask))
        endpoints = np.unique(endpoints)
        c += 1
        if c > 1000:
            break
    
    angles_dict = getAngles(all_vertices, endpoints)
    endpoints = filterEndpoints(endpoints, angles_dict, len(all_vertices))
    angles_list = list(getAngles(all_vertices, endpoints).keys())
    endpoints = np.sort(np.asarray(angles_list)[:num])
    
    piecewiseCurves = []
    for i in range(len(endpoints)):
        if i == len(endpoints) - 1:
            p1 = all_vertices[endpoints[i]:all_vertices.shape[0]-1]; p2 = all_vertices[0:endpoints[0]+1]
            piece = np.concatenate((p1, p2))
        else:
            piece = all_vertices[int(endpoints[i]):int(endpoints[(i+1)%len(endpoints)])+1]
        piecewiseCurves.append(piece)
    
    if len(piecewiseCurves) > 0:
        while(len(piecewiseCurves)) < num:
            longest = -1; pos = 0
            for i in range(len(piecewiseCurves)):
                if longest < piecewiseCurves[i].shape[0]:
                    longest = piecewiseCurves[i].shape[0]
                    pos = i

            piece = piecewiseCurves.pop(pos)
            p1 = np.copy(piece[:piece.shape[0]//2]); p2 = np.copy(piece[piece.shape[0]//2:])
            piecewiseCurves.insert(i, p1); piecewiseCurves.insert(i+1, p2)

    return piecewiseCurves


# # t-Init Function

# In[11]:


def initializeTNumpy(piece, alpha=1):
    t = np.zeros(piece.shape[0])
    for j in range(1, piece.shape[0]):
        t[j] = t[j-1] + np.linalg.norm(piece[j-1] - piece[j]) ** alpha
    return t / t[-1]


# # Reconstruct From Param

# In[12]:


def func(x, coef):
    out = None
    for i in range(len(coef)):
        if out is None:
            out = coef[i] * x**i
        else:
            out += coef[i] * x**i            
    return out


# In[13]:


def reconstruct(parameterized, step=0.1, t=None): 
    # parameterized: shape = (num_pieces x 3, curve degree)
    
    if t is None:
        t = np.linspace(0, 1, num=int(1/step))
    
    points = None
    
    for i in range(0, parameterized.shape[0], 3):
        x_t = func(t, parameterized[i+0,:])
        y_t = func(t, parameterized[i+1,:])        
        z_t = func(t, parameterized[i+2,:])
        
        if points is None:
            points = np.array((x_t,y_t,z_t))
        else:
            points = np.concatenate((points, (x_t,y_t,z_t)), axis=1)
            
    return points


# In[14]:


def makeA_b(pieces, degree=4):
    dim_0 = 0
    for p in range(len(pieces)):
        dim_0 += pieces[p].shape[0]
    A = np.zeros((dim_0, (degree+1)*len(pieces)))
    
    prev_index = 0
    bx, by, bz = None, None, None
    for p in range(len(pieces)):
        p1_x, p1_y, p1_z = pieces[p][:,0], pieces[p][:,1], pieces[p][:,2]
        t1 = initializeTNumpy(pieces[p])
        
        A1 = np.ones((t1.shape[0],1))
        for i in range(1,degree+1):
            A1 = np.concatenate((A1, t1.reshape((t1.shape[0],1))**i), axis=1)
        A[prev_index:prev_index+A1.shape[0],p*(degree+1):(p+1)*(degree+1)] = A1
        prev_index += np.copy(A1.shape[0])
        
        if bx is None:
            bx = np.copy(p1_x.reshape(p1_x.shape[0],1))
            by = np.copy(p1_y.reshape(p1_y.shape[0],1))
            bz = np.copy(p1_z.reshape(p1_z.shape[0],1))
        else:
            bx = np.concatenate((bx, np.copy(p1_x.reshape(p1_x.shape[0],1))))
            by = np.concatenate((by, np.copy(p1_y.reshape(p1_y.shape[0],1))))
            bz = np.concatenate((bz, np.copy(p1_z.reshape(p1_z.shape[0],1))))
    
    return A, bx, by, bz


# In[15]:


def makeC_d(pieces, degree=4):
    C = np.zeros(((len(pieces))*2, len(pieces)*(degree+1)))
    for p in range(len(pieces)):
        if p == len(pieces) - 1:
            C[2*p,p*(degree+1):(p+1)*(degree+1)] = 1; C[2*p,(p+1)%len(pieces)*(degree+1)] = -1
            C[2*p+1,p*(degree+1)+1:(p+1)*(degree+1)] = np.asarray([i for i in range(1, degree+1)])
            C[2*p+1,(p+1)%len(pieces)*(degree+1)+1] = -1
        else:
            C[2*p,p*(degree+1):(p+1)*(degree+1)] = 1; C[2*p,(p+1)*(degree+1)] = -1
            C[2*p+1,p*(degree+1)+1:(p+1)*(degree+1)] = np.asarray([i for i in range(1, degree+1)])
            C[2*p+1,(p+1)*(degree+1)+1] = -1
    
    return C, np.zeros((C.shape[0],1))


# In[16]:


def makeC2_d(pieces, degree=4):
    C = np.zeros(((len(pieces))*3, len(pieces)*(degree+1)))
    for p in range(len(pieces)):
        if p == len(pieces)-1:
            C[3*p,p*(degree+1):(p+1)*(degree+1)] = 1; C[3*p,(p+1)%len(pieces)*(degree+1)] = -1
            C[3*p+1,p*(degree+1)+1:(p+1)*(degree+1)] = np.asarray([i for i in range(1, degree+1)]); C[3*p+1,(p+1)%len(pieces)*(degree+1)+1] = -1
            C[3*p+2,p*(degree+1)+2:(p+1)*(degree+1)] = np.asarray([i*(i-1) for i in range(2, degree+1)]); C[3*p+2,(p+1)%len(pieces)*(degree+1)+2] = -2
        else:
            C[3*p,p*(degree+1):(p+1)*(degree+1)] = 1; C[3*p,(p+1)*(degree+1)] = -1
            C[3*p+1,p*(degree+1)+1:(p+1)*(degree+1)] = np.asarray([i for i in range(1, degree+1)]); C[3*p+1,(p+1)*(degree+1)+1] = -1
            C[3*p+2,p*(degree+1)+2:(p+1)*(degree+1)] = np.asarray([i*(i-1) for i in range(2, degree+1)]); C[3*p+2,(p+1)*(degree+1)+2] = -1
        
    return C, np.zeros((C.shape[0],1))


# In[ ]:


def resample_cross_section_random(mesh, sample_num=1.0e3):
    points = None
    
    # Change radius according to the mesh bounds / diagonals
    min_bounds, max_bounds = mesh.bounds
    diagonal = np.linalg.norm(max_bounds - min_bounds)
    
    param_list = []
    param_list_only = []

    # for i in tqdm.trange(int(sample_num)):
    i = 0
    pbar = tqdm.tqdm(total = sample_num)

    while i < int(sample_num):
        centre = mesh.centroid
        
        # Point on unit sphere
        theta = np.random.rand() * np.pi * 2
        phi = np.random.rand() * np.pi
        
        x = diagonal / 8 * np.sin(phi) * np.cos(theta)
        y = diagonal / 8 * np.sin(phi) * np.sin(theta)
        z = diagonal / 8 * np.cos(phi)
        
        point = centre + np.array([x,y,z])
        
        # Normal
        normal = np.random.randn(3)
        normal = normal / np.linalg.norm(normal)
        
        # Get Slice
        slices = mesh.section(plane_origin=point, plane_normal=normal)
        if slices is None:
            continue

        all_vertices = np.array(slices.vertices)
        
        # Parameter Dictionary
        skip = False
        
        all_params = {}
        all_params['normal'] = normal
        all_params['point'] = point
        all_params['array'] = None
        
        all_params_only = None

        per_entity_point = np.array([len(slices.entities[t].points) for t in range(len(slices.entities))])
        ratio = np.copy(per_entity_point) / np.sum(per_entity_point)
        
        num_per_entity = num * ratio
        num_per_entity = np.floor(num_per_entity).astype(np.longlong)
        if np.sum(num_per_entity) < num:
            num_per_entity[np.argmin(num_per_entity)] += num - np.sum(num_per_entity)
        if np.sum(num_per_entity) != num:
            print('Failed')
            sys.exit()
        
        print(num_per_entity, np.sum(num_per_entity), len(slices.entities))
        
        for j in range(len(slices.entities)):
            if num_per_entity[j] == 0:
                continue

            order = slices.entities[j].points
            curr_vertices = all_vertices[order]

            pieces = getPiecewiseCurvesAdaptive(curr_vertices, num=num_per_entity[j])
            l = len(pieces)
            
            # Flag Check to ensure required number of pieces returned
            if l < num_per_entity[j]:
                print(l)
                skip = True
            
            # Flag Check to ensure each piece has sufficient points
            for p in range(len(pieces)):
                if len(pieces[p]) < degree + 1:
                    print(pieces[p].shape)
                    skip = True
            
            if skip == True:
                break
            
            C_test, d = makeC_d(pieces, degree)
            zero_mat_size = 2

            A_test, bx, by, bz = makeA_b(pieces, degree)

            nA1 = np.concatenate((np.matmul(A_test.T, A_test), C_test.T), axis=1)
            nA2 = np.concatenate((C_test, np.zeros((C_test.shape[0],(l)*zero_mat_size))), axis=1)
            nA = np.concatenate((nA1, nA2), axis=0)

            nB = np.concatenate((np.matmul(A_test.T, bx), d), axis=0); x = np.matmul(np.linalg.inv(nA), nB)
            nB = np.concatenate((np.matmul(A_test.T, by), d), axis=0); y = np.matmul(np.linalg.inv(nA), nB)
            nB = np.concatenate((np.matmul(A_test.T, bz), d), axis=0); z = np.matmul(np.linalg.inv(nA), nB)
            
            for it in range(l):
                params = np.array((x[it*(degree+1):it*(degree+1)+(degree+1)], 
                           y[it*(degree+1):it*(degree+1)+(degree+1)], 
                           z[it*(degree+1):it*(degree+1)+(degree+1)]))
                if all_params_only is None:
                    all_params['array'] = np.copy(params)
                    all_params_only = np.copy(params)
                else:
                    all_params['array'] = np.concatenate((all_params['array'], np.copy(params)), axis=0)
                    all_params_only = np.concatenate((all_params_only, np.copy(params)), axis=0)
        
        if not skip:
            param_list_only.append(all_params_only)
            param_list.append(all_params)
            i = i + 1
            pbar.update(1)
    pbar.close()
                    
    return param_list, param_list_only


# # Final Code

# In[ ]:


def make_dataset_mnt(root='./ShapeNet', target='./ShapeNet2', subset=None, 
                     step=None, trimap=False, normalized=False, uniform=False):
    sub_folders = os.listdir(root)
    sub_folders.sort()
    
    path_list = []
    
    if not os.path.exists(target):
        os.makedirs(target)
    
    for i in tqdm.trange(len(sub_folders)):
        folder = sub_folders[i]
        sub_sub_folders = os.listdir(os.path.join(root, folder))
        sub_sub_folders.sort()
        
        if not os.path.exists(os.path.join(target, folder)):
            os.makedirs(os.path.join(target, folder))
        
        if subset is not None:
            sub_index = np.random.permutation(len(sub_sub_folders))[:subset]
        else:
            sub_index = np.arange(len(sub_sub_folders))
        
        for tempj in tqdm.trange(len(sub_index)):
            j = sub_index[tempj]
            sub = sub_sub_folders[j]
            curr_folder = os.path.join(root, folder, sub, 'models')
            curr_target = os.path.join(target, folder, sub, 'models')
            
            if not os.path.exists(os.path.join(target, folder, sub)):
                os.makedirs(os.path.join(target, folder, sub))
                
            if not os.path.exists(curr_target):
                os.makedirs(curr_target)

            files = os.listdir(curr_folder)
            files.sort()

            if 'model_manifold.obj' in files:
                path = os.path.join(curr_folder, 'model_manifold.obj')
                path_list.append((path, curr_target))
                
    return path_list


def get_mesh_splits(mesh, num_patches=5):

    patches = [mesh]

    sample_points = []

    while len(patches)!=num_patches:
        mesh = patches.pop()
        p_, n_ = split_mesh(mesh)
#         print('split done')
    #     verts, faces  = get_verts_faces(p_)
    #     p_ = trimesh.Trimesh(verts, faces)
    #     verts, faces = get_verts_faces(n_)
    #     n_ = trimesh.Trimesh(verts, faces)

        patches.insert(0, p_)
        patches.insert(0, n_)
    return patches


def split_mesh(mesh):
    mesh_points = mesh.vertices
    done = False
    while not done:
        plane_normal_ = np.random.randn(3)
        plane_origin_ = np.mean(mesh_points, 0)
        mesh_pos = trimesh.intersections.slice_mesh_plane(mesh, plane_normal_, plane_origin_)
        mesh_neg = trimesh.intersections.slice_mesh_plane(mesh, -plane_normal_, plane_origin_)
        
        if mesh_pos.vertices.shape[0] > 0 and mesh_neg.vertices.shape[0]>0:
            done = True
            break
    return mesh_pos, mesh_neg


def return_fixed_mesh(mesh):
    meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    meshfix.repair()
    
    mesh_fixed = trimesh.Trimesh(meshfix.v, meshfix.f)
    
    return mesh_fixed
    
    
def make_param_CS(path):
    mesh = read_mesh(path[0])
    
    mesh_patches = get_mesh_splits(mesh, 3)

    random_patch = random.choice(mesh_patches)
    fixed_patch = return_fixed_mesh(random_patch)
    
    mesh = fixed_patch
                
    # if os.path.exists(os.path.join(curr_target, 'model_list.pth')):
    #     continue

    min_x, min_y, min_z = mesh.bounds[0]
    max_x, max_y, max_z = mesh.bounds[1]
    mesh.vertices[:,0] = (mesh.vertices[:,0] - min_x) / (max_x - min_x)
    mesh.vertices[:,1] = (mesh.vertices[:,1] - min_y) / (max_y - min_y)
    mesh.vertices[:,2] = (mesh.vertices[:,2] - min_z) / (max_z - min_z)

    pl, pl2 = resample_cross_section_random(mesh, sample_num=100)

    pickle.dump(pl, open(os.path.join(path[1], 'model_dict.pth'), 'wb'))
    pickle.dump(pl2, open(os.path.join(path[1], 'model_list.pth'), 'wb'))


# In[ ]:


path_list = make_dataset_mnt(root='/home/aradhya/Surface_Reconstruction/ShapeNetCore.v2/', 
                 target='/mnt/part1/aradhya/ShapeNet_params_num1000_cs100_C1_fast_final_hope_arb', 
                 normalized=True, subset=20)

print('generating...')
with mp.Pool(20) as p:
        list(tqdm.tqdm(p.imap(make_param_CS, path_list), total=len(path_list)))

print('done')