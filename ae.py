import os
import sys
import tqdm
import torch
import datetime

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from pyntcloud import PyntCloud
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from tqdm import tqdm

torch.__version__

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = [nn.Conv1d(3, 64, kernel_size=1), 
                nn.BatchNorm1d(64),
                nn.ReLU()]
        conv2 = [nn.Conv1d(64, 128, kernel_size=1), 
                nn.BatchNorm1d(128),
                nn.ReLU()]
        conv3 = [nn.Conv1d(128, 256, kernel_size=1), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        conv4 = [nn.Conv1d(256, 128, kernel_size=1), 
                nn.BatchNorm1d(128),
                nn.AdaptiveMaxPool1d(1)]
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)        
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        
        print('initialising encoder')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
#         print(out_1.shape)
#         print(out_2.shape)
#         print(out_3.shape)
#         print(out_4.shape)
        out_4 = out_4.view(-1, out_4.shape[1])
#         print(out_4.shape)
        return out_4

# e = Encoder().to(device)
# e.forward(torch.randn(1,3,2048).to(device))

class Decoder(nn.Module):
    def __init__(self, num_points):
        super(Decoder, self).__init__()
        linear1 = [nn.Linear(128, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        linear2 = [nn.Linear(256, 256), 
                nn.BatchNorm1d(256),
                nn.ReLU()]
        linear3 = [nn.Linear(256, 6144), 
                ]
        self.linear1 = nn.Sequential(*linear1)
        self.linear2 = nn.Sequential(*linear2)
        self.linear3 = nn.Sequential(*linear3)
        self.num_points = num_points
        print('initialising decoder')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
                print(m)
                torch.nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        out_1 = self.linear1(x)
        out_2 = self.linear2(out_1)
        out_3 = self.linear3(out_2)
        return out_3.view(-1, 3, self.num_points)

class AutoEncoder(nn.Module):
    def __init__(self, num_points):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_points)
        
        
    def forward(self, x):
        gfv = self.encoder(x)
        out = self.decoder(gfv)
        
        return out, gfv

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) :
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        

class ChamferLoss(nn.Module):
    def __init__(self, num_points):
        super(ChamferLoss, self).__init__()
        self.num_points = num_points
        self.loss = torch.FloatTensor([0]).to(device)

        
    def forward(self, predict_pc, gt_pc):
        z, _ = torch.min(torch.norm(gt_pc.unsqueeze(-2) - predict_pc.unsqueeze(-1),dim=1), dim=-2)
            # self.loss += z.sum()
        self.loss = z.sum() / (len(gt_pc)*self.num_points)

        z_2, _ = torch.min(torch.norm(predict_pc.unsqueeze(-2) - gt_pc.unsqueeze(-1),dim=1), dim=-2)
        self.loss += z_2.sum() / (len(gt_pc)*self.num_points)
        return self.loss




class PointcloudDatasetAE(Dataset):
    def __init__(self, root, list_point_clouds):
        self.root = root
        self.list_files = list_point_clouds
        
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        points = PyntCloud.from_file(self.list_files[index])
        points = np.array(points.points)
        points_normalized = (points - (-0.5)) / (0.5 - (-0.5))
        points = points_normalized.astype(float)
        points = torch.from_numpy(points)
        
        return points

def plot(out_data, points, name, num_samples=5):
#     for i in range(num_samples):
#     import ipdb; ipdb.set_trace()
    output = out_data[0,:,:]
    output = output.permute([1,0]).detach().cpu().numpy()

    inputt = points[0,:,:]
    inputt = inputt.detach().cpu().numpy()

    fig = plt.figure()
    ax_x = fig.add_subplot(111, projection='3d')
    x_ = output
    ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
    ax_x.set_xlim([0,1])
    ax_x.set_ylim([0,1])
    ax_x.set_zlim([0,1])
    fig.savefig(name + '_{}_out.png'.format(i))

    fig = plt.figure()
    ax_x = fig.add_subplot(111, projection='3d')
    x_ = inputt
    ax_x.scatter(x_[:, 0], x_[:, 1], x_[:,2])
    ax_x.set_xlim([0,1])
    ax_x.set_ylim([0,1])
    ax_x.set_zlim([0,1])
    fig.savefig(name + '_{}_in.png'.format(i))

    plt.close('all')
 


def get_dataloaders(DATA_DIR = '../data/shape_net_core_uniform_samples_2048/'):
    

    folders = os.popen('ls '+DATA_DIR)
    folders = folders.read().split("\n")[:-1]
    snc_synth_id_to_category = {
        '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
        '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
        '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
        '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
        '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
        '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
        '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
        '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
        '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
        '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
        '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
        '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
        '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
        '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
        '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
        '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
        '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
        '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
        '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
    }
    required = set(['airplane', 'chair', 'lamp', 'sofa'])

    point_cloud_paths = []
    for folder in folders:
        if snc_synth_id_to_category[folder] in required:
            files = os.popen('ls '+DATA_DIR+"/"+folder).read().split("\n")[:-1]
            files = [DATA_DIR+folder+"/"+file for file in files]
            point_cloud_paths.append(files)

    all_point_cloud_paths = []
    for cloud_set in point_cloud_paths:
        all_point_cloud_paths += cloud_set
    print(len(all_point_cloud_paths))

    pc_list = []
    for pc_path in all_point_cloud_paths:
        mesh = trimesh.load(pc_path)
        pc_np = np.array(mesh.vertices)
        pc_list.append(pc_np)
    pc_numpy = np.array(pc_list)

    pc_numpy.shape

    # list_point_clouds = np.load('./list_point_subset.npy')
    # list_point_clouds = list_point_clouds[:5000]
    list_point_clouds = all_point_cloud_paths

    np.save('point_clouds.npy', list_point_clouds)
    
    print(len(list_point_clouds))
    X_train, X_test, _, _ = train_test_split(list_point_clouds, list_point_clouds, test_size=0.05, random_state=42)
    print(len(X_train))



    train_dataset = PointcloudDatasetAE(DATA_DIR, X_train)
    train_dataloader = DataLoader(train_dataset, num_workers=8, shuffle=False, batch_size=40, pin_memory=True)

    test_dataset = PointcloudDatasetAE(DATA_DIR, X_test)
    test_dataloader = DataLoader(test_dataset, num_workers=2, shuffle=False, batch_size=1)
    return train_dataloader, test_dataloader, X_train, X_test

if __name__ == '__main__':

    autoencoder = AutoEncoder(2048).to(device)
    chamfer_loss = ChamferLoss(2048).to(device)
#     autoencoder.apply(weights_init)
    
#     vertices = pc_numpy[0].reshape(3,2048)
#     ax_old = plt.axes(projection ="3d")
#     ax_old.scatter3D(vertices[0,:], vertices[1,:], vertices[2,:], color = "green")
#     plt.show()

    lr = 1.0e-4
    momentum = 0.9

    optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(momentum, 0.999))

    train_dataloader, test_dataloader, X_train, X_test = get_dataloaders()


    test_mode = False

    ROOT_DIR = './ae_out_copy/'
    now =   str(datetime.datetime.now())
    now += "initialised"
    
    if test_mode:
        now += '_test'
    
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

    EVAL_OUTPUTS_DIR = ROOT_DIR  + now + '/eval_outputs/'
    if not os.path.exists(EVAL_OUTPUTS_DIR):
        os.makedirs(EVAL_OUTPUTS_DIR)

    MODEL_DIR = ROOT_DIR + now + '/models/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    summary_writer = SummaryWriter(LOG_DIR)





    for epoch in tqdm(range(0,1000)):

        for i, data in enumerate(train_dataloader):

            autoencoder.train()
            optimizer_AE.zero_grad()

            data = data.permute([0,2,1]).float().to(device)

            out_data, gfv = autoencoder(data)

            loss = chamfer_loss(out_data, data) 
            loss.backward()
            optimizer_AE.step()
            if test_mode:
                print(i, loss.item())


            if i % (len(train_dataloader) // 100)  == 0:
                print('Epoch: {}, Iteration: {}/{}, Content Loss: {}'.format(epoch, i,len(train_dataloader), loss.item()))
                summary_writer.add_scalar('ch_loss', loss.item(), global_step=epoch*len(train_dataloader)+i)

            if test_mode and i == 5:
                break
    #         if i > 2:
    #             break

    #     break
        plot(out_data, data.permute(0,2,1), name=OUTPUTS_DIR+'train_outputs_{}'.format(epoch))
        torch.save(autoencoder.state_dict(), MODEL_DIR+'{}_ae_.pt'.format(epoch))
        if test_mode :
            break
    # autoencoder = AutoEncoder(2048).to(device)
    # autoencoder.load_state_dict(torch.load('./ae_out_copy/2019-11-27 01:31:11.870769/models/990_ae_.pt')) # Edit this to add the correct path
    # eval_dir = './eval_output/'
    # if not os.path.exists(eval_dir):
    #     os.makedirs(eval_dir)


    eval_losses = []
    for i in range(len(X_test)):
            points = PyntCloud.from_file(X_test[i])
            points = np.array(points.points)
#             points_normalized = points 
    #         points_normalized = (points - points.min()) / (points.max() - points.min())
            points_normalized = (points - (-0.5)) /  (0.5 - (-0.5)) # uncomment if points are unnormalized
            points = points_normalized.astype(np.float)
            points = torch.from_numpy(points).unsqueeze(0)
            points = points.permute([0,2,1]).float().to(device)
    #         print(points.shape)

            autoencoder.eval()

            with torch.no_grad():
                    out_data, gfv = autoencoder(points)
                    loss = chamfer_loss(out_data, points)


            eval_losses.append(loss.item())

            plot(out_data, points.permute(0,2,1), name=EVAL_OUTPUTS_DIR+'eval_outputs_{}'.format(epoch), num_samples=1)

            if test_mode:
                break
    summary_writer.add_scalar('eval_loss',np.average(eval_losses))
    print('Avg Eval Loss', np.average(eval_losses))

    summary_writer.close()