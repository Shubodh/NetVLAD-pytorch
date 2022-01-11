import torch
import torch.nn as nn
from torch.autograd import Variable

from netvlad import NetVLAD
from netvlad import EmbedNet
from hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18
from scipy.spatial.distance import cdist

import cv2
import numpy as np
import glob

import sys

def getMatchInds(ft_ref,ft_qry,topK=1,metric='cosine'):
    """
    This function takes two matrics and computes the distance between every vector in the matrices.
    For every query vector a ref. vector having shortest distance is retured
    """
    """
    metric: 'euclidean' or 'cosine' or 'correlation'
    """
    dMat = cdist(ft_ref,ft_qry,metric)
    mInds = np.argsort(dMat,axis=0)[:topK]        # shape: K x ft_qry.shape[0]
    return mInds

# Discard layers at the end of base network
encoder = resnet18(pretrained=True)
#print(encoder)
base_model = nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)
dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
num_clusters=32
net_vlad = NetVLAD(num_clusters=num_clusters, dim=dim, alpha=1.0)
#print(f"descriptor dimension, num_clusters: {dim, num_clusters}")
#print(f"final vector dimension would be: dim * num_clusters: {dim * num_clusters}")
model = EmbedNet(base_model, net_vlad).cuda()

# Define loss
criterion = HardTripletLoss(margin=0.1).cuda()

# This is just toy example. Typically, the number of samples in each classes are 4.
#labels = torch.randint(0, 10, (40, )).long()
base_path = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/SOTA_repos_NetVLAD-LoFTR-PatchNetVLAD-etc/all_NetVLAD/NetVLAD-pytorch/sample_graphVPR_data/"
base_rooms = ['01', '02', '03', '04']
images1, images2, images3, images4 = [], [], [], []
images_all = [images1, images2, images3, images4]
for i, base_room in enumerate(base_rooms):
    for img in glob.glob(base_path + base_room + "/*.jpg"):
        rgb= cv2.imread(img)
        rgb_np = np.moveaxis(np.array(rgb), -1, 0)
        rgb_np = rgb_np[np.newaxis, :]
        x = torch.from_numpy(rgb_np).float().cuda()
        output = model(x)
        images_all[i].append(output)
#print((images_all[0].shape))
roomVect1 = images_all[0][0] + images_all[0][1] + images_all[0][2]
roomVect2 = images_all[1][0] + images_all[1][1] + images_all[1][2]
roomVect3 = images_all[2][0] + images_all[2][1] + images_all[2][2]
roomVect4 = images_all[3][0] + images_all[3][1] + images_all[3][2]
dim = (roomVect1.cpu().detach().numpy().shape[-1])
featVect = np.zeros((4, dim))
featVect[0] = roomVect1.cpu().detach().numpy()
featVect[1] = roomVect2.cpu().detach().numpy()
featVect[2] = roomVect3.cpu().detach().numpy()
featVect[3] = roomVect4.cpu().detach().numpy()
mInds1 = getMatchInds(featVect, featVect, topK=2)
print(mInds1)
#rgb_np = np.moveaxis(np.array(rgb), -1, 0)
#rgb_np = rgb_np[np.newaxis, :]
#print(rgb_np.shape)
#x = torch.rand(40, 3, 128, 128).cuda()
#x = torch.from_numpy(rgb_np).float().cuda()
#output = model(x)
#triplet_loss = criterion(output, labels)

#print(f"input, output.shape {x.shape, output}")
#print(f"labels: {labels}")
#print(f"triplet_loss {triplet_loss}")
print("successfully reached end")