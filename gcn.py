import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GATConv

import torch_geometric
from torch_geometric.data import Dataset as GDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as GDataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb

import tqdm
import pickle
import trimesh
import os
import random
from sklearn.model_selection import train_test_split
from itertools import cycle

import matplotlib.pyplot as plt

seed_val = 123

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)

def func(x, coef):
    out = None
    for i in range(len(coef)):
        if out is None:
            out = coef[i] * x**i
        else:
            out += coef[i] * x**i            
    return out


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


# # Dataloader

# In[ ]:


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


# In[ ]:

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

def mesh_patch_split_points(mesh, num_patches=5, num_samples=1000):
    patches  = get_mesh_splits(mesh, num_patches)
    sample_points = []
    for patch in patches:
        samples, _ = trimesh.sample.sample_surface(patch, num_samples)
        sample_points.append(samples)
    
    return np.array(sample_points)



def get_c_adjacency(num_cross, batch_size):
    adj = []
    for k in range(batch_size):
        per_adj = []
        for i in range(num_cross):
            for j in range(num_cross):
                if i != j:
                    per_adj.append([i+k*num_cross,j+k*num_cross])
                    per_adj.append([j+k*num_cross,i+k*num_cross])
                    # per_adj.append([i+num_cross,j+num_cross])
                    # per_adj.append([i+num_cross,j+num_cross])
        adj.append(per_adj)
    return adj

# def get_p_adjacency(num_pieces, batch_size,start=0):
#     adjacencies = []
#     u_adj = []
#     for i in range(start, start+num_pieces-1):
#         u_adj.append([i,i+1])
#         u_adj.append([i+1,i])
#     u_adj.append([start+num_pieces-1, 0])
#     u_adj.append([0,start+num_pieces-1])
#     u_adj = np.array(u_adj)
#     # print(u_adj.shape)
#     u_adj = u_adj.reshape(1, -1, 2)
#     # print(u_adj.shape)
#     u_adj = torch.from_numpy(u_adj).repeat(batch_size,1, 1)
    
#     for i in range(batch_size):
#         u_adj[i] += i * num_pieces
    
#     return u_adj.reshape(-1,2)

def get_adjacency(num_pieces,start=0):
    adjacencies = []
    u_adj = []
    for i in range(start, start+num_pieces-1):
        u_adj.append([i,i+1])
        u_adj.append([i+1,i])
    u_adj.append([start+num_pieces-1, 0])
    u_adj.append([0,start+num_pieces-1])

    return u_adj

def get_p_adjacency_fixed(num_pieces, num_cross, batch_size=1, start=0):
    adjacencies = []
    # u_adj = []
    for b in range(batch_size):
        u_adj = []
        for k in range(num_cross):
            start = k * num_pieces
            end = start + num_pieces
            # print(start, end)
            for i in range(start, end-1):
                u_adj.append([i, i+1])
                u_adj.append([i+1, i])

            u_adj.append([end-1, 0])
            u_adj.append([0, end-1])
        u_adj = np.array(u_adj)
        u_adj += b * num_cross * num_pieces
        adjacencies.append(u_adj)
    u_adj = np.array(adjacencies).reshape(-1,2)
    return u_adj

def get_p_all_adj(num_pieces, num_cross, batch_size=1, start=0):
    adjacencies = []
    u_adj = []
    start=0
    for b in range(batch_size):
        for n in range(0,num_cross):
            start = n * num_pieces + b * (num_cross * num_pieces)
            end = n * num_pieces + num_pieces + b * (num_cross * num_pieces)
            # print(start, end)
            for i in range(start, end):
                for j in range(start, end):
                    if i!=j:
                        u_adj.append([i,j])
                        u_adj.append([j,i])
        
    return np.array(u_adj)


class MDset(GDataset):
    def __init__(self, root_mesh, root_pieces, phase='train', input_num=20, 
                 random_input=True, single_obj=True, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.root_mesh = root_mesh; self.root_pieces = root_pieces
        self.input_num = input_num
        self.random_input = random_input
        self.single_obj = single_obj
        
        self.list_volumes = self.get_all_volumes()
        self.train_volumes, _, self.test_volumes, _ = train_test_split(self.list_volumes, 
                                                                       self.list_volumes, 
                                                                       random_state=seed_val)
        self.train_volumes, _, self.val_volumes, _ = train_test_split(self.train_volumes, 
                                                                      self.train_volumes, 
                                                                      random_state=seed_val)
        self.num_patches = 16
        self.num_sample_points = 128
        if phase == 'train':
            self.volumes = self.train_volumes
        elif phase == 'val':
            self.volumes = self.val_volumes
        else:
            self.volumes = self.test_volumes
    @property
    def raw_file_names(self):
        return self.list_volumes

    @property
    def processed_file_names(self):
        return self.train_volumes

    
    def get_all_volumes(self):
        list_volumes = []
        sub_folders = os.listdir(self.root_pieces)
        sub_folders.sort()
        sub_folders = ['03001627', '02691156', '03636649', '04256520']
        # chair, airplane , lamp, sofa
        
        # sub_folders = ['03001627'] # chair
        
        for i in tqdm.trange(len(sub_folders)):
            folder = sub_folders[i]
            sub_sub_folders = os.listdir(os.path.join(self.root_pieces, folder))
            sub_sub_folders.sort()
            
            sub_index = np.arange(len(sub_sub_folders))
            
            for j in sub_index:
                sub = sub_sub_folders[j]
                curr_folder_model = os.path.join(self.root_mesh, folder, sub, 'models')
                curr_folder_cross = os.path.join(self.root_pieces, folder, sub, 'models')
                
                files = os.listdir(curr_folder_cross)
                files.sort()
            
                for f in files:
                    if 'list' in f:
                        path_model = os.path.join(curr_folder_model, 'model_manifold.obj')
                        path_cross = os.path.join(curr_folder_cross, f)
                        
                        list_volumes.append((path_model, path_cross))
                        
            if self.single_obj:
                break
        
        return list_volumes
                

    def len(self):
            return len(self.processed_file_names)

    def get(self, index):
        f = pickle.load(open(self.volumes[index][1], 'rb'))
        pieces = torch.from_numpy(np.array(f))
        
        if self.random_input:
            # print('generating random input')
            # exit()
            input_set = np.random.permutation(pieces.shape[0])[:self.input_num]
        else:
            input_set = np.arange(0,self.input_num)
        input_pieces = pieces[input_set]
        idx = np.random.choice(range(self.input_num), np.random.randint(5,self.input_num), replace=False) #minimum 5 cross-sections
        input_pieces = input_pieces[idx]
        mesh = read_mesh(self.volumes[index][0])
        # print(self.volumes[index][0])
        input_pieces = input_pieces.permute(3,0,1,2)
        # input_pieces = input_pieces.reshape(1, input_pieces.shape[1])
        input_pieces_ = input_pieces.reshape(input_pieces.shape[0], input_pieces.shape[1], input_pieces.shape[2]//3, -1)

        points,_ = trimesh.sample.sample_surface(mesh, self.num_patches * self.num_sample_points)
        p_adj = torch.from_numpy(get_p_adjacency_fixed(input_pieces_.shape[-2],input_pieces_.shape[1])).long()
        # p_adj = torch.from_numpy(get_p_all_adj(input_pieces_.shape[-2], input_pieces_.shape[1])).long()
        data = Data(input_pieces_.reshape(-1,18).float(), edge_index=p_adj.t().contiguous())
        # data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return input_pieces,data, torch.from_numpy(points).float()



class EncoderS(nn.Module):
    def __init__(self, num_points=2048, global_feat=True):
        super(EncoderS, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 512, 1)
        # self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, 128, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(512)
        # self.bn4 = nn.BatchNorm1d(256)

        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(
                    m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
                if m.bias is not None:
                    m.bias.data.zero_()  # initialize bias as zero
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.bn1(F.relu((self.conv1(x))))
        pointfeat = x
        x = self.bn2(F.relu((self.conv2(x))))
        x = self.bn3(F.relu((self.conv3(x))))
        x = self.bn4(F.leaky_relu((self.conv4(x))))
        x = self.bn5(F.leaky_relu((self.conv5(x))))
        x = self.mp1(x)
        # print(x.shape)
        x = x.view(batchsize, 512)
        # if self.global_feat:
        # print(x.shape)
        return x
class MyLinear(nn.Module):
    def __init__(self, indim, odim, activation, normalization=None):
        super().__init__()
        self.linear = nn.Linear(indim, odim)
        self.activation = activation
        self.normalization = nn.BatchNorm1d(num_features=odim)
    def forward(self, x):
        out = self.linear(x)
        if self.activation:
            out = self.activation(out)
        # print(out.shape, self.normalization)
        # if self.normalization:
        #     out = self.normalization(out.unsqueeze(1)).squeeze(1)
        return out
class DecoderS(nn.Module):
    def __init__(self):
        super(DecoderS, self).__init__()
        
        self.linear1 = MyLinear(512, 1024, activation=nn.LeakyReLU(), normalization=nn.BatchNorm1d(num_features=256))
        self.linear2 = MyLinear(1024, 1024, activation=nn.LeakyReLU(), normalization=nn.BatchNorm1d(num_features=512))
        self.linear_out = MyLinear(1024, 6144, activation=None, normalization=None)

        # special initialization for linear_out, to get uniform distribution over the space
        self.linear_out.linear.bias.data.uniform_(-1, 1)
    
    def forward(self, x):
        # reshape from feature vector NxC, to NxC
        # print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear_out(x)

        return x.view(-1, 3, 2048)


class AE_pointnet(nn.Module):
    def __init__(self, num_points=2048,global_feat= False):
        super(AE_pointnet, self).__init__()
        self.encoder = EncoderS(num_points = num_points, global_feat = global_feat)
        self.decoder = DecoderS()

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = torch.transpose(x,1,2)
        encoder = self.encoder(x)
        # print(encoder.shape)
        decoder = self.decoder(encoder)


        return decoder

# class EncoderS(nn.Module):
#     def __init__(self, num_points=2048, global_feat=True):
#         super(EncoderS, self).__init__()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 256, 1)
#         self.conv4 = torch.nn.Conv1d(256, 128, 1)

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(128)

#         self.mp1 = torch.nn.MaxPool1d(num_points)
#         self.num_points = num_points
#         self.global_feat = global_feat

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(
#                     m, nn.Linear):
#                 torch.nn.init.kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
#                 if m.bias is not None:
#                     m.bias.data.zero_()  # initialize bias as zero
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.bn4(self.conv4(x))
#         x = self.mp1(x)
#         # print(x.shape)
#         x = x.view(batchsize, 128)
#         # if self.global_feat:
#         return x
# class MyLinear(nn.Module):
#     def __init__(self, indim, odim, activation, normalization=None):
#         super().__init__()
#         self.linear = nn.Linear(indim, odim)
#         self.activation = activation
#         self.normalization = nn.BatchNorm1d(num_features=odim)
#     def forward(self, x):
#         out = self.linear(x)
#         if self.activation:
#             out = self.activation(out)
#         # print(out.shape, self.normalization)
#         # if self.normalization:
#         #     out = self.normalization(out.unsqueeze(1)).squeeze(1)
#         return out
# class DecoderS(nn.Module):
#     def __init__(self):
#         super(DecoderS, self).__init__()
        
#         self.linear1 = MyLinear(128, 256, activation=nn.LeakyReLU(), normalization=nn.BatchNorm1d(num_features=256))
#         self.linear2 = MyLinear(256, 256, activation=nn.LeakyReLU(), normalization=nn.BatchNorm1d(num_features=256))
#         self.linear_out = MyLinear(256, 6144, activation=None, normalization=None)

#         # special initialization for linear_out, to get uniform distribution over the space
#         self.linear_out.linear.bias.data.uniform_(-1, 1)
    
#     def forward(self, x):
#         # reshape from feature vector NxC, to NxC
#         # print(x.shape)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear_out(x)

#         return x.view(-1, 3, 2048)


class AE_pointnetEnc(nn.Module):
    def __init__(self, num_points=2048,global_feat= False):
        super().__init__()
        self.encoder = EncoderS(num_points = num_points, global_feat = global_feat)
        self.decoder = DecoderS()

    def forward(self, x):
        x = torch.squeeze(x,dim=1)
        x = torch.transpose(x,1,2)
        emb = self.encoder(x)
        # print(encoder.shape)
        # decoder = self.decoder(emb)


        return emb


class AE_pointnetDec(nn.Module):
    def __init__(self, num_points=2048,global_feat= False):
        super().__init__()
        self.encoder = EncoderS(num_points = num_points, global_feat = global_feat)
        self.decoder = DecoderS()

    def forward(self, emb):
        # x = torch.squeeze(x,dim=1)
        # x = torch.transpose(x,1,2)
        # emb = self.encoder(x)
        # print(encoder.shape)
        # print(emb.shape, self.decoder.shape)
        decoded = self.decoder(emb)


        return decoded

class GCN(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = SAGEConv(18, dim)
        # self.bn1 = torch_geometric.nn.BatchNorm(dim)
        # self.bn1 = torch_geometric.nn.DiffGroupNorm(dim, 32)
        self.bn1 = torch_geometric.nn.PairNorm(dim)
        
        self.conv2 = SAGEConv(dim, dim*4)
        # self.bn2 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn2 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        self.bn2 = torch_geometric.nn.PairNorm(dim*4)
        
        self.conv21 = SAGEConv(dim*4, dim*4)
        # self.bn21 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn21 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        
        self.conv22 = SAGEConv(dim*4, dim*4)
        # self.bn22 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn22 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        self.bn21 = torch_geometric.nn.PairNorm(dim*4)
        
        self.conv3 = SAGEConv(dim*4, dim*4)
        # self.bn3 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn3 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        self.bn3 = torch_geometric.nn.PairNorm(dim*4)
        self.linear11 = nn.Sequential(*[nn.Linear(dim*4, dim*4),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*4, dim*4),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*4,dim*4)
                                        ])
        # self.linear111 = nn.Linear(dim*4, dim*4)
        # self.linear112 = nn.Linear(dim*4, dim*4)
        self.linear12 = nn.Sequential(*[nn.Linear(dim*8, dim*8),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*8, dim*8),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*8,dim*8)
                                        ])
        self.gat_conv1 = GATConv(dim*4,dim*4)
        self.p_pool = nn.AdaptiveAvgPool1d(1)
        self.c_pool = nn.AdaptiveAvgPool1d(1)
        self.gat_conv2 = GATConv(dim*4,dim*8)
        # self.linear1 = nn.Linear(1,3)
        self.linear1 = nn.Linear(4,3)
        self.linear2 = nn.Sequential(*[nn.Linear(128, 64),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.3),
                                       nn.Linear(64, 128),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.3),
                                       nn.Linear(128, 512),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.3),
                                       # nn.Linear(256, 128),
                                       # nn.LeakyReLU(), 
                                       # nn.Dropout(0.3),
                                      ])
        self.linear3 = nn.Linear(512, 2048)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout21 = nn.Dropout(0.3)
        self.dropout22 = nn.Dropout(0.3)
        # self.global_attention = GlobalAttention(1, num_cross, num_params)
        self.dim = dim
    def forward(self, data, batch_size, num_cross, num_pieces):
        x, edge_index = data.x.float(), data.edge_index
        # x += torch.randn_like(x).float()
        
        x = self.bn1(F.leaky_relu(self.conv1(x, edge_index)))
        # print(edge_index.shape)
        # x = (x)
        # x = F.dropout(x, training=self.training)
        x = self.dropout1(self.bn2(F.leaky_relu(self.conv2(x, edge_index))))
        x = self.dropout21(self.bn21(F.leaky_relu(self.conv21(x, edge_index))))
        x = self.dropout22(self.bn22(F.leaky_relu(self.conv22(x, edge_index))))
        x = self.dropout2(self.bn3(F.leaky_relu(self.conv3(x, edge_index))))
        p_adj_all = torch.from_numpy(get_p_all_adj(num_pieces, num_cross)).long().t().contiguous().to(device)
        x = self.gat_conv1(x, p_adj_all)
        x = x.reshape(batch_size, num_cross, num_pieces, -1)
        # x = x.mean(-2) 
        x = x.permute(0,1,3,2)
        shape = x.shape
        x = self.p_pool(x.reshape(shape[0], shape[1]*shape[2], shape[3]))
        x = x.squeeze(-1)
        x = x.reshape(shape[0], shape[1], shape[2])
        x = F.leaky_relu(self.linear11(x))
        
        # print(x.shape)
        # edge_index1 = torch.ones(2, 100)
        edge_index1 = torch.tensor(get_c_adjacency(num_cross, batch_size)).long().t().contiguous().to(device)
        # print(edge_index.shape, edge_index1.shape, x.shape)
        x = self.gat_conv2(x.reshape(num_cross * batch_size , self.dim*4), edge_index1)
        x = x.reshape(batch_size, num_cross, -1)#.mean(1)
        x = x.permute(0,2,1)
        shape = x.shape
        x = self.c_pool(x)
        x = x.squeeze(-1)
        x = F.leaky_relu(self.linear12(x))
        # x = x + torch.randn_like(x).to(device)
        # x = self.global_attention(x)
        x = x.reshape(x.shape[0], 4, 128)
        x = F.leaky_relu(self.linear2(x))
        # print(x.shape)
        x = F.leaky_relu(self.linear1(x.permute(0,2,1)))
        # print(x.shape)
        x = F.tanh(self.linear3(x.permute(0,2,1))) * 2.
        return x.permute(0,2,1)



class GCNEmb(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = SAGEConv(19, dim)
        self.bn1 = torch_geometric.nn.BatchNorm(dim)
        # self.bn1 = torch_geometric.nn.DiffGroupNorm(dim, 32)
        
        self.conv2 = SAGEConv(dim, dim*4)
        self.bn2 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn2 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        
        
        self.conv21 = SAGEConv(dim*4, dim*4)
        self.bn21 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn21 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        
        self.conv22 = SAGEConv(dim*4, dim*4)
        self.bn22 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn22 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        
        
        self.conv3 = SAGEConv(dim*4, dim*4)
        self.bn3 = torch_geometric.nn.BatchNorm(dim*4)
        # self.bn3 = torch_geometric.nn.DiffGroupNorm(dim*4, 32)
        self.linear11 = nn.Sequential(*[nn.Linear(dim*4, dim*4),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*4, dim*4),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*4,dim*4)
                                        ])
        # self.linear111 = nn.Linear(dim*4, dim*4)
        # self.linear112 = nn.Linear(dim*4, dim*4)
        self.linear12 = nn.Sequential(*[nn.Linear(dim*8, dim*8),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*8, dim*8),
                                         nn.LeakyReLU(),
                                         nn.Linear(dim*8, 512),
                                         nn.LeakyReLU()
                                        ])
        self.gat_conv1 = GATConv(dim*4,dim*4)
        self.p_pool = nn.AdaptiveAvgPool1d(1)
        self.c_pool = nn.AdaptiveAvgPool1d(1)
        self.gat_conv2 = GATConv(dim*4,dim*8)
        # self.linear1 = nn.Linear(1,3)
        # self.linear1 = nn.Linear(4,3)
        # self.linear2 = nn.Sequential(*[nn.Linear(128, 64),
        #                                nn.LeakyReLU(),
        #                                nn.Dropout(0.3),
        #                                nn.Linear(64, 128),
        #                                nn.LeakyReLU(),
        #                                nn.Dropout(0.3),
        #                                nn.Linear(128, 512),
        #                                nn.LeakyReLU(),
        #                                nn.Dropout(0.3),
        #                                # nn.Linear(256, 128),
        #                                # nn.LeakyReLU(), 
        #                                # nn.Dropout(0.3),
        #                               ])
        # self.linear3 = nn.Linear(512, 2048)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout21 = nn.Dropout(0.3)
        self.dropout22 = nn.Dropout(0.3)
        # self.global_attention = GlobalAttention(1, num_cross, num_params)
        self.dim = dim
    def forward(self, data, batch_size, num_cross, num_pieces):
        x, edge_index = data.x.float(), data.edge_index
        # x += torch.randn_like(x).float()
        
        x = self.bn1(F.leaky_relu(self.conv1(x, edge_index)))
        # print(edge_index.shape)
        # x = (x)
        # x = F.dropout(x, training=self.training)
        x = self.dropout1(self.bn2(F.leaky_relu(self.conv2(x, edge_index))))
        x = self.dropout21(self.bn21(F.leaky_relu(self.conv21(x, edge_index))))
        x = self.dropout22(self.bn22(F.leaky_relu(self.conv22(x, edge_index))))
        x = self.dropout2(self.bn3(F.leaky_relu(self.conv3(x, edge_index))))
        
        ''' Made change here '''
        # p_adj_all = torch.from_numpy(get_p_all_adj(num_pieces, num_cross)).long().t().contiguous().to(device)
        # x = self.gat_conv1(x, p_adj_all)

        x = self.gat_conv1(x, edge_index)

        x = x.reshape(batch_size, num_cross, num_pieces, -1)
        # x = x.mean(-2) 
        x = x.permute(0,1,3,2)
        shape = x.shape
        x = self.p_pool(x.reshape(shape[0], shape[1]*shape[2], shape[3]))
        x = x.squeeze(-1)
        x = x.reshape(shape[0], shape[1], shape[2])
        x = F.leaky_relu(self.linear11(x))
        
        # print(x.shape)
        # edge_index1 = torch.ones(2, 100)
        edge_index1 = torch.tensor(get_c_adjacency(num_cross, batch_size)).long().t().contiguous().to(device)
        # print(edge_index.shape, edge_index1.shape, x.shape)
        x = self.gat_conv2(x.reshape(num_cross * batch_size , self.dim*4), edge_index1)
        x = x.reshape(batch_size, num_cross, -1)#.mean(1)
        x = x.permute(0,2,1)
        shape = x.shape
        x = self.c_pool(x)
        x = x.squeeze(-1)
        x = F.leaky_relu(self.linear12(x))

        return x

# class GCNDisc(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(18, 32)
#         self.conv2 = GCNConv(32, 64)
#         self.conv3 = GCNConv(64, 128)
#         self.gat_conv1 = GATConv(128,128)
#         self.gat_conv2 = GATConv(128,128)
#         # self.linear1 = nn.Linear(1,3)
#         self.linear2 = nn.Linear(128, 128)
#         self.linear3 = nn.Linear(128, 1)
#         # self.dropout1 = nn.Dropout(0.3)
#         # self.dropout2 = nn.Dropout(0.3)
#         # self.global_attention = GlobalAttention(1, num_cross, num_params)
        
#     def forward(self, data, batch_size, num_cross, num_pieces):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         # print(edge_index.shape)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = F.leaky_relu(self.conv2(x, edge_index))
#         x = F.leaky_relu(self.conv3(x, edge_index))
#         x = self.gat_conv1(x, edge_index)
#         x = x.reshape(batch_size, num_cross, num_pieces, -1)
#         x = x.mean(-2)
#         # print(x.shape)
#         # edge_index1 = torch.ones(2, 100)
#         edge_index1 = torch.tensor(get_c_adjacency(num_cross, batch_size)).long().t().contiguous()
#         # print(edge_index.shape, edge_index1.shape, x.shape)
#         x = self.gat_conv2(x.reshape(num_cross * batch_size , 128), edge_index1)
#         x = x.reshape(batch_size, num_cross, -1).mean(1)
        
#         # x = self.global_attention(x)
#         x = x.reshape(x.shape[0],  -1)
#         x = F.leaky_relu(self.linear2(x))
#         x = F.sigmoid(self.linear3(x))
#         # x = F.sigmoid(self.linear1(x.permute(0,2,1)))
        
#         return x

class GCNDisc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(19, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)
        self.gat_conv1 = GATConv(64,64)
        self.gat_conv2 = GATConv(64,64)
        self.linear1 = nn.Linear(1,3)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.pts_model = nn.Sequential(*[
                    nn.Conv1d(3, 32, 3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(32),
                    nn.Conv1d(32, 64, 3, stride=2, padding=1),
                    nn.LeakyReLU(),    
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 64, 3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.AdaptiveAvgPool1d(1)
            
        ]) 
        
        # self.global_attention = GlobalAttention(1, num_cross, num_params)
        
    def forward(self, data, points, batch_size, num_cross, num_pieces):
        
        x, edge_index = data.x.float(), data.edge_index
        # x += torch.randn_like(x).float()
        x = self.conv1(x, edge_index)
        # print(edge_index.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.gat_conv1(x, edge_index)
        x = x.reshape(batch_size, num_cross, num_pieces, -1)
        x = x.mean(-2)
        # print(x.shape)
        # edge_index1 = torch.ones(2, 100)
        edge_index1 = torch.tensor(get_c_adjacency(num_cross, batch_size)).long().t().contiguous().to(device)
        # print(edge_index.shape, edge_index1.shape, x.shape)
        x = self.gat_conv2(x.reshape(num_cross * batch_size , 64), edge_index1)
        x = x.reshape(batch_size, num_cross, -1).mean(1)
        
        # x = self.global_attention(x)
        x = x.reshape(x.shape[0],  -1)
        pts_out = self.pts_model(points)
        # print(pts_out.shape, x.shape)
        x += pts_out.squeeze(-1)
        x = F.leaky_relu(self.linear2(x))
        
        
        x = F.sigmoid(self.linear3(x))
        # x = F.sigmoid(self.linear1(x.permute(0,2,1)))
        
        
        return x


class GCNEmbDisc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(19, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)
        self.gat_conv1 = GATConv(64,64)
        self.gat_conv2 = GATConv(64,64)
        
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(256, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.pts_model = nn.Sequential(*[
                    nn.Conv1d(3, 32, 3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(32),
                    nn.Conv1d(32, 64, 3, stride=2, padding=1),
                    nn.LeakyReLU(),    
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 64, 3, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.AdaptiveAvgPool1d(1)
            
        ]) 
        self.emb_linear = nn.Linear(512, 128)
        # self.global_attention = GlobalAttention(1, num_cross, num_params)
        
    def forward(self, data, emb, batch_size, num_cross, num_pieces):
        
        x, edge_index = data.x.float(), data.edge_index
        # x += torch.randn_like(x).float()
        x = self.conv1(x, edge_index)
        # print(edge_index.shape)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.gat_conv1(x, edge_index)
        x = x.reshape(batch_size, num_cross, num_pieces, -1)
        x = x.mean(-2)
        # print(x.shape)
        # edge_index1 = torch.ones(2, 100)
        edge_index1 = torch.tensor(get_c_adjacency(num_cross, batch_size)).long().t().contiguous().to(device)
        # print(edge_index.shape, edge_index1.shape, x.shape)
        x = self.gat_conv2(x.reshape(num_cross * batch_size , 64), edge_index1)
        x = x.reshape(batch_size, num_cross, -1).mean(1)
        x = F.leaky_relu(self.linear2(x))
        # print(x.shape)
        emb = F.leaky_relu(self.emb_linear(emb))
        x = torch.cat([x, emb], -1)
        # print(x.shape)
#         # x = self.global_attention(x)
#         x = x.reshape(x.shape[0],  -1)
#         pts_out = self.pts_model(points)
#         # print(pts_out.shape, x.shape)
#         x += pts_out.squeeze(-1)
#         x = F.leaky_relu(self.linear2(x))
        
        
#         x = F.sigmoid(self.linear3(x))
        x = F.sigmoid(self.linear3(x))
        
        
        return x
    
    

class ChamferLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ChamferLoss, self).__init__()
        self.loss = torch.FloatTensor([0]).to(device)
        self.reduction = reduction
    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) - predict_pc.unsqueeze(-1),dim=1), dim=-2)
            # self.loss += z.sum()
        if self.reduction != 'sum':
            self.loss = z.sum() / (len(gt_pc))
        else:
            self.loss = z.sum()
        z_2, _ = torch.min(torch.norm(predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1),dim=1), dim=-2)
        if self.reduction != 'sum':
            self.loss += z_2.sum() / (len(gt_pc))
        else:
            self.loss += z_2.sum()
        return self.loss
    
def save_sample( pt, name):
    # for p in range(1):
    p = 0
    pts = pt[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2])

    plt.savefig(name+'_'+str(p)+'.png', format = "png")
    # plt.show()
    print('saved', name+'_'+str(p)+'.png')
    plt.close()
    
if __name__ == '__main__':
    device = torch.device('cuda')

    n_b = 1
    n_c = 10
    # single_mesh = True
    num_epoch = 1000
    num_gen = 5

    num_patches = 16
    num_sample_points = 128

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    root_mesh = './../../ShapeNetCore.v2/'
    root_pieces = '/mnt/part1/aradhya//ShapeNet_params_all_cs100_C1_fast_final_please/'

    gdset = MDset(root_mesh, root_pieces, input_num=n_c, single_obj=False)
    g_dloader = GDataLoader(gdset, batch_size=n_b)

    # min_ = 99999.
    # max_ = -99999.
    # for pieces,batch_g,pts in tqdm.tqdm((g_dloader)):
    #     break
    #     if pts.min().item() < min_:
    #         min_ = pts.min().item()
    #     if pts.max().item() > max_:
    #         max_ = pts.max().item()
    # print(min_, max_)
    # exit()



    params = {'g_lr': 1e-4, 'd_lr': 1e-5}
    # dim = 64 # 64 default
    dim = 128
    gcn = GCNEmb(dim).to(device)
    gcn_disc = GCNEmbDisc().to(device)
    pt_enc = AE_pointnetEnc().to(device)
    pt_dec = AE_pointnetDec().to(device)

    pt_enc.eval()
    pt_dec.eval()

    tags = ['multiclass', 'embedding', 'adaptive', 'random_input', 'pairnorm', 'dim_{}'.format(dim)] #'continued', 'aircraft'
    notes = 'embedding concat noise {} all models nc {} nb {} linear cgan all no chamfer pairnorm'.format(tags[0], n_c, n_b)

    ROOT_DIR = './graph_conv_gan_multilvl_adj_{}_embedding_concat_adaptive_random_input_bn/'.format(tags[0])
    now = 'graph_conv_noise_in_big_all_connected_chamfer_clipped_grad_2000_n_b_{}_n_c{}_dim_{}_pairnorm'.format(n_b, n_c, dim) 
    #################################
    # gcn.load_state_dict(torch.load(os.path.join(ROOT_DIR,'graph_conv_noise_in_big_all_connected_chamfer_clipped_grad_2000_n_b_1_n_c102022-01-10 01:12:27.996067', 'models', 'new_test_dict_gen_int.pth' )))
    # gcn_disc.load_state_dict(torch.load(os.path.join(ROOT_DIR,'graph_conv_noise_in_big_all_connected_chamfer_clipped_grad_2000_n_b_1_n_c102022-01-10 01:12:27.996067', 'models', 'new_test_dict_disc_int.pth' )))
    print(tags)
    if tags[0] == 'aircraft':
        print('aircraft model loaded')
        pt_enc.load_state_dict(torch.load('')) # aircraft models
        pt_dec.load_state_dict(torch.load(''))
    elif tags[0] == 'chair':
        print('chair model loaded')
        pt_enc.load_state_dict(torch.load('')) # chair models
        pt_dec.load_state_dict(torch.load(''))
    elif tags[0] == 'multiclass':
        print('loading multiclass model.... ')
        pt_enc.load_state_dict(torch.load(''))
        pt_dec.load_state_dict(torch.load(''))
        print('loaded', tags[0], 'models')
    
    print('########## models loaded ###########')

    optimizerG = torch.optim.Adam(gcn.parameters(), lr=params['g_lr'])
    optimizerD = torch.optim.Adam(gcn_disc.parameters(), lr=params['d_lr'])

    criterion_points = ChamferLoss()
    criterion_gan = nn.BCELoss()

    epochs = 2000
    test_mode = False

    max_ch = 9999999.
    from datetime import datetime


    now +=  str(datetime.now())

    if test_mode:
        now += '_test'
        notes += '_test'
        
    wandb.init(project='graph_conv_multi_adj_embedding', entity='aradhya', config=params, notes=notes, tags=tags)
    wandb.watch((gcn, gcn_disc), log='all', log_graph=True, log_freq=100)

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    if not os.path.exists(ROOT_DIR + now):
        os.makedirs(ROOT_DIR + now)

    LOG_DIR = ROOT_DIR + now + '/logs/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    OUTPUTS_DIR = ROOT_DIR  + now + '/outputs/'
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    MODEL_DIR = ROOT_DIR + now + '/models/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    summary_writer = SummaryWriter(LOG_DIR)


    with open(os.path.join(MODEL_DIR,'model_g.txt'), 'w+') as f:
        f.writelines(str(gcn))

    with open(os.path.join(MODEL_DIR,'model_d.txt'), 'w+') as f:
        f.writelines(str(gcn_disc))

        
    start_time = datetime.now()
    display_st = start_time.strftime("%H:%M:%S")

    


    try:
        for epoch in range(epochs):
            epoch_ch = []
            for i, (pieces,batch_g,pts) in tqdm.tqdm(enumerate(g_dloader), total=len(g_dloader)):  #, total=len(g_dloader))
                if torch.isnan(pieces).any() or torch.isinf(pieces).any() :
                    continue
                gcn.train()
                gcn_disc.train()
                pieces = pieces.squeeze(1).to(device)
                
                batch_g.x =  torch.cat([batch_g.x,torch.randn(batch_g.x.shape[0], 1)],1) 
                batch_g = batch_g.to(device)
                # print(batch_g.x.shape)
                # exit()
                pts = pts.to(device).float()
                # print(pts.shape)
                # exit()
                with torch.no_grad():
                    emb_gt = pt_enc(pts)
                
                truth = torch.ones(pieces.shape[0], 1).float().to(device)
                fake = torch.zeros(pieces.shape[0], 1).float().to(device)
            
                
                emb_pred = gcn(batch_g, pieces.shape[0], pieces.shape[1], pieces.shape[2]//3)
                
                with torch.no_grad():
                    points_pred = pt_dec(emb_pred)
                    # print(points_pred.shape, '############3')
                    # exit()
                
                optimizerD.zero_grad()
                
                real_prob = gcn_disc(batch_g, emb_gt, pieces.shape[0], pieces.shape[1], pieces.shape[2]//3)
                fake_prob = gcn_disc(batch_g, emb_pred, pieces.shape[0], pieces.shape[1], pieces.shape[2]//3)
                
                # print(real_prob, fake_prob)
                # try:
                loss_real = criterion_gan(real_prob, truth)
                loss_fake = criterion_gan(fake_prob, fake)
                loss_dis = (loss_real + loss_fake) / 2
                # loss_dis *= 10.
                # except:
                #     import ipdb; ipdb.set_trace()
                # if loss_dis.item() > 0.5:
                loss_dis.backward()
                optimizerD.step()
                
                for _ in range(2):
                    optimizerG.zero_grad()
                    emb_pred = gcn(batch_g, pieces.shape[0], pieces.shape[1], pieces.shape[2]//3)
                    
                    fake_prob = gcn_disc(batch_g, emb_pred, pieces.shape[0], pieces.shape[1], pieces.shape[2]//3) 
                    loss_gan = criterion_gan(fake_prob, truth) 
                    with torch.no_grad():
                        points_pred = pt_dec(emb_pred)
                        points_gt = pt_dec(emb_gt)
                        
                    emb_loss = F.mse_loss(emb_pred, emb_gt, reduction='sum')

                    # print(points_pred.shape, pts.shape)
                    # ch_loss = criterion_points(points_pred, pts.permute(0,2,1))
                    ch_loss = criterion_points(points_pred, points_gt)
                    if ch_loss.item() > 600.:

                        print(i, points_pred.min(), points_pred.max(), pts.min(), pts.max())
                        # exit()
                    loss_gen = loss_gan  + ch_loss + emb_loss 

                    loss_gen.backward()
                    # torch.nn.utils.clip_grad_norm(gcn.parameters(), 10.)

                    optimizerG.step()
                
                if i % 10 == 0:
                    print()
                    print('start time', display_st)
                    print('current time', datetime.now().strftime("%H:%M:%S"))
                    # print('time elapsed', (datetime.now()-start_time).strftime("%H:%M:%S"))
                    print('Epoch', epoch, 'iter', i, 'Batches', len(g_dloader))
                    print('loss d real ', loss_real.item())
                    print('loss d fake ', loss_fake.item())
                    print('loss d', loss_dis.item())
                    print()
                    print(pieces.shape)
                    print(batch_g.x.shape)
                    print(points_pred.min(), points_pred.max(), pts.min(), pts.max())
                    print('loss g ', loss_gan.item())
                    print('Emb loss', emb_loss.item())
                    print('chamfer loss ', ch_loss.item())
                    print('loss G total ', loss_gen.item())
                    print(pts.shape, points_pred.shape)
                    print(emb_gt.shape, emb_pred.shape)
                    print()
                    
                epoch_ch.append(ch_loss.item())
                
                wandb.log({'loss d real ': loss_real.item()})
                wandb.log({'loss d fake ': loss_fake.item()})
                wandb.log({'loss d': loss_dis.item()})
                
                wandb.log({'emb_loss': emb_loss.item()})
                wandb.log({'loss g ': loss_gan.item()})
                wandb.log({'chamfer loss ': ch_loss.item()})
                wandb.log({'loss G total' : loss_gen.item()})
                

                summary_writer.add_scalar('loss d real ', loss_real.item())
                summary_writer.add_scalar('loss d fake ', loss_fake.item())
                summary_writer.add_scalar('loss d', loss_dis.item())
                
                summary_writer.add_scalar('emb_loss', emb_loss.item())
                summary_writer.add_scalar('loss g ', loss_gan.item())
                summary_writer.add_scalar('chamfer loss ', ch_loss.item())
                summary_writer.add_scalar('loss G total' , loss_gen.item())
                

                if  (i % (len(g_dloader) // 4) == 0):
                    print('Saving samples to ', os.path.join(OUTPUTS_DIR, '{}_{}_{}_gt'.format(epoch, i, pieces.shape[1])))
                    save_sample(pts.cpu(), os.path.join(OUTPUTS_DIR, '{}_{}_{}_gt'.format(epoch, i, pieces.shape[1])))
                    save_sample(points_pred.permute(0,2,1).cpu().detach(), os.path.join(OUTPUTS_DIR, '{}_{}_{}_out'.format(epoch, i, pieces.shape[1])))
                    save_sample(points_gt.permute(0,2,1).cpu().detach(), os.path.join(OUTPUTS_DIR, '{}_{}_{}_gt_gen'.format(epoch, i, pieces.shape[1])))
                
                if test_mode:
                    break
            # epoch_ch = 
            print('Average CH:', np.average(epoch_ch))
            wandb.log({'avg ch': np.average(epoch_ch)})

            if np.average(epoch_ch) < max_ch:
                max_ch = np.average(epoch_ch)
                torch.save(gcn.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_gen_best.pth'))
                torch.save(gcn_disc.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_disc_best.pth'))
                
            if test_mode:
                break
            
            torch.save(gcn.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_gen_latest.pth'))
            torch.save(gcn_disc.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_disc_latest.pth'))
    except KeyboardInterrupt:

            torch.save(gcn.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_gen_int.pth'))
            torch.save(gcn_disc.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_disc_int.pth'))


    torch.save(gcn.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_gen_last.pth'))
    torch.save(gcn_disc.state_dict(), os.path.join(MODEL_DIR, './new_test_dict_disc_last.pth'))
