from PIL import Image
from tensorboard.compat.proto import tensor_pb2
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_dir = "hymenoptera_data\\train\\bees\\16838648_415acd9e3f.jpg"
img = Image.open(img_dir)
writer = SummaryWriter('logs')
#ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)

#Normalize
img_normalize = transforms.Normalize([0.7,0.9,0.1],[0.5,0.5,0.5])
img_nor = img_normalize(img_tensor)
writer.add_image('Normalize_image1', img_nor, 0)
writer.close()