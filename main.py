import glob
import torch
import tensorly as tl
from torch import nn, optim
from skimage import data, filters
import os
import skimage
from tqdm import tqdm
from utils import *
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import math
import time
import cv2
import torch
import random
tl.set_backend('pytorch')
image_path = ""  # /home/zzy/sirst/train/images/000006.png
save_path = ""
label_path = ''
torch.cuda.set_device(2)
expand_x = 1
n_3 = 500
expand = 1
shrink = 5

# gamma 是超参数 可调
gamma = 0.000000001

max_iter = 2500#可调
lr_real = 0.1#学习率，可调
last = 0
start = last
end = 80
plus = start

class Y_net(nn.Module):
    def __init__(self, n_1, n_2, n_3):  #
        super(Y_net, self).__init__()
        self.A_hat = nn.Parameter(torch.Tensor(n_3 * expand, n_1, n_2 // shrink))  # // shrink
        self.B_hat = nn.Parameter(torch.Tensor(n_3 * expand, n_2 // shrink, n_2))
        self.net = nn.Sequential(
            nn.Conv3d(1,1, kernel_size=(3, 3,3), padding='same'),nn.BatchNorm3d(1),
            nn.LeakyReLU(),
            nn.Conv3d(1,1, kernel_size=(3, 3,3), padding='same'),
            nn.Linear(int(80), 80, bias=False))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A_hat.size(0))
        self.A_hat.data.uniform_(-stdv, stdv)  # 连续分布范围
        self.B_hat.data.uniform_(-stdv, stdv)

    def forward(self):
        x = torch.matmul(self.A_hat, self.B_hat)
        x = x.reshape(1,1,x.shape[1], x.shape[2], x.shape[0])
        return self.net(x)



dir = image_path
imgList = os.listdir(dir)
imgList.sort(key=lambda x: int(x.split('.')[0]))
a = cv2.imread(os.path.join(dir, '0.bmp'), 0)
m, n = a.shape
img = torch.zeros(m, n, 80)
for count in range(0, 80):
    im_name = imgList[count + last]
    im_path = os.path.join(dir, im_name)
    img[:, :, count] = torch.from_numpy(cv2.imread(im_path, 0)) / 255.0  # 0
X = img.cuda()
model = Y_net(X.shape[0], X.shape[1], X.shape[2]).cuda()  #
mask = torch.ones(X.shape).cuda()
mask[X == 0] = 0
X[mask == 0] = 0
params = []
params += [x for x in model.parameters()]
s = sum([np.prod(list(p.size())) for p in params])
print('Number of params: %d' % s)
optimizier = optim.Adam(params, lr=lr_real, weight_decay=10e-8)  # 10e-8
t0 = time.time()

for iter in range(max_iter):
    # F_norm = nn.MSELoss()
    X_Out_real = model()
    c = torch.norm(torch.squeeze(X_Out_real)*mask-X * mask, 1)
    b1 = torch.norm(tl.unfold(torch.squeeze(X_Out_real) * mask, mode=0), p='nuc')
    b2 = torch.norm(tl.unfold(torch.squeeze(X_Out_real) * mask, mode=1), p='nuc')
    b3 = torch.norm(tl.unfold(torch.squeeze(X_Out_real) * mask, mode=2), p='nuc')
    loss = b1 + b2 + b3 +0.2*c # 超参可调

    i = 0
    for p in params:
        i += 1
    if i == 1:  # A
       loss += gamma * torch.norm(p[:, 1:, :] - p[:, :-1, :], 1)

    if i == 5:  # H_k
       loss += gamma * torch.norm(p[1:,:] - p[:-1,:], 1)

    if i == 2:  # B
       loss += gamma * torch.norm(p[:, :, 1:] - p[:, :, :-1], 1)
    optimizier.zero_grad()
    loss.backward(retain_graph=True)

    optimizier.step()
    t1 = time.time()

img_real = X * mask - torch.squeeze(X_Out_real) * mask
img_real2 = img_real.cpu().detach().numpy() * 255.0
maxvalue = np.max(img_real2)
thresh = 0.5 * maxvalue
img_real2[img_real2 >= thresh] = 255
img_real2[img_real2 <thresh] = 0
for j in range(start, end):# 存储
    img_name1 = str('{:0=3}'.format(j))
    save_path1 = save_path + '/' + img_name1 + '.png'
    imgs = img_real2[:, :, j - plus]
    cv2.imwrite(save_path1, imgs)


print(t1 - t0)



