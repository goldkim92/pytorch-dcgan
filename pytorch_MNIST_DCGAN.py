import os, time
import matplotlib.pyplot as plt
import scipy.misc as scm
import itertools
import pickle
import imageio
from tqdm import tqdm
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
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input))) # 8d*4*4
        x = F.relu(self.deconv2_bn(self.deconv2(x))) # 4d*8*8
        x = F.relu(self.deconv3_bn(self.deconv3(x))) # 2d*16*16
        x = F.relu(self.deconv4_bn(self.deconv4(x))) # d*32*32
        x = F.tanh(self.deconv5(x)) # 1*64*64
        return x

    
class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
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
        x = F.leaky_relu(self.conv1(input), 0.2) # d*32*32
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2) # 2d*16*16
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2) # 4d*8*8
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2) # 8d*4*4
        x = F.sigmoid(self.conv5(x))

        return x

    
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((8 * 8, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)


def show_result(count, isFix=False):
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
    img_grid = 0.5*(img_grid[0,:,:] + 1.)
    # image summary
    writer.add_image('image', img_grid, count)
    scm.imsave(os.path.join(args.sample_dir, '{:05d}.png'.format(count)), img_grid)

    
# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print('training start!')
start_time = time.time()
count = 0
for epoch in range(train_epoch):
    print('Epoch[{}/{}]'.format(epoch+1, train_epoch))
    
    for x_, _ in tqdm(train_loader):
        count += 1
        ''' 
        train discriminator D 
        '''
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        D_real_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_real_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)

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

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
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
