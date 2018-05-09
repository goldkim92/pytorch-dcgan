import os, time, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as scm
import itertools
import pickle
import imageio
from tqdm import tqdm
from glob import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from tensorboardX import SummaryWriter


# argument parser
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_number',     type=str,   default='0')
parser.add_argument('--data_dir',       type=str,   default=os.path.join('.','data','celebA'))
parser.add_argument('--log_dir',        type=str,   default='log') # in assets/ directory
parser.add_argument('--ckpt_dir',       type=str,   default='checkpoint') # in assets/ directory
parser.add_argument('--sample_dir',     type=str,   default='sample') # in assets/ directory
parser.add_argument('--test_dir',       type=str,   default='test') # in assets/ directory
parser.add_argument('--assets_dir',     type=str,   default=None,   required=True) # if assets_dir='aa' -> assets_dir='./assets/aa'

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

assets_dir = os.path.join('.','assets',args.assets_dir)
args.log_dir = os.path.join(assets_dir, args.log_dir)
args.ckpt_dir = os.path.join(assets_dir, args.ckpt_dir)
args.sample_dir = os.path.join(assets_dir, args.sample_dir)
args.test_dir = os.path.join(assets_dir, args.test_dir)

# make directory if not exist
try: os.makedirs(args.log_dir)
except: pass
try: os.makedirs(args.ckpt_dir)
except: pass
try: os.makedirs(args.sample_dir)
except: pass
try: os.makedirs(args.test_dir)
except: pass

# summary writer
writer = SummaryWriter(args.log_dir)


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x2 = F.relu(self.deconv1_bn(self.deconv1(input))) # 8d*4*4
        x3 = F.relu(self.deconv2_bn(self.deconv2(x2))) # 4d*8*8
        x4 = F.relu(self.deconv3_bn(self.deconv3(x3))) # 2d*16*16
        x5 = F.relu(self.deconv4_bn(self.deconv4(x4))) # d*32*32
        x6 = F.tanh(self.deconv5(x5)) # 3*64*64

        return x6, x3

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x5 = F.leaky_relu(self.conv1(input), 0.2) # d*32*32
        x4 = F.leaky_relu(self.conv2_bn(self.conv2(x5)), 0.2) # 2d*16*16
        x3 = F.leaky_relu(self.conv3_bn(self.conv3(x4)), 0.2) # 4d*8*8
        x2 = F.leaky_relu(self.conv4_bn(self.conv4(x3)), 0.2) # 8d*4*4
        x1 = F.sigmoid(self.conv5(x2))

        return x1
    
    # feature forward method
    def feature_forward(self, feature):
        x2 = F.leaky_relu(self.conv4_bn(self.conv4(feature)), 0.2) # 8d*4*4
        x1 = F.sigmoid(self.conv5(x2))

        return x1
    

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((8 * 8, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)

def show_result(count, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    # Generate batch of images and convert to grid
    img_grid = make_grid(test_images.cpu().data)
    img_grid = 0.5*(img_grid + 1.)
    # image summary
    writer.add_image('image', img_grid, count)
#    scm.imsave(os.path.join(args.sample_dir, '{:05d}.png'.format(count)), img_grid)


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

'''
celebA preprocessing & post-processing
'''
    
def load_data_list(data_dir):
    path = os.path.join(data_dir, 'train', '*')
    file_list = glob(path)
    return file_list

def preprocess_image(file_list, input_size, phase='train'):
    imgA = [get_image(img_path, input_size, phase=phase) for img_path in file_list]
    return np.array(imgA)

def get_image(img_path, input_size, phase='train'):
    img = scm.imread(img_path) # 218*178*3
    img_crop = img[34:184,14:164,:] #188*160*3
    img_resize = scm.imresize(img_crop,[input_size,input_size,3])
    img_resize = img_resize/127.5 - 1.
    
    if phase == 'train' and np.random.random() >= 0.5:
        img_resize = np.flip(img_resize,1)
	
    img_resize = np.transpose(img_resize, [2,0,1])    
    return img_resize

def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)

# data_loader
file_list = load_data_list(args.data_dir)
batch_idxs = len(file_list) // batch_size

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# D & G feature update parameters
Gf_parameter = [
    {'params': G.deconv1.parameters()},
    {'params': G.deconv1_bn.parameters()},
    {'params': G.deconv2.parameters()},
    {'params': G.deconv2_bn.parameters()}
]
Df_parameter = [
    {'params': D.conv4.parameters()},
    {'params': D.conv4_bn.parameters()},
    {'params': D.conv5.parameters()}
]

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
Gf_optimizer = optim.Adam(Gf_parameter, lr=lr, betas=(0.5, 0.999))
Df_optimizer = optim.Adam(Df_parameter, lr=lr, betas=(0.5, 0.999))

print('Training start!')
start_time = time.time()
count = 0
for epoch in range(train_epoch):
    print('Epoch[{}/{}]'.format(epoch+1, train_epoch))
    
    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
        
    for idx in tqdm(range(batch_idxs)):
        count += 1
        
        # get batch images and labels
        x_ = preprocess_image(file_list[idx*batch_size:(idx+1)*batch_size], input_size=64)
        x_ = torch.from_numpy(x_).float()
        
        '''
        train discriminator D
        '''
        D.zero_grad()

        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        D_real_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_real_result, y_real_)

        z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)[0]

        D_fake_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_fake_result, y_fake_)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()
        
        # summary the real_d & fake_d value
        writer.add_scalars('val',{'real_d':D_real_result.mean(), 'fake_d':D_fake_result.mean()}, count)
        
        '''
        train generator G
        '''
        G.zero_grad()

        z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)[0]
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()
        
        # summary D_loss & G_loss
        writer.add_scalars('loss',{'D':D_train_loss, 'G':G_train_loss}, count)
        
        if count % 100 == 0:
            show_result(count, isFix=True)
            torch.save(G.state_dict(), os.path.join(args.ckpt_dir,"generator_param.pkl"))
            torch.save(D.state_dict(), os.path.join(args.ckpt_dir,"discriminator_param.pkl"))
