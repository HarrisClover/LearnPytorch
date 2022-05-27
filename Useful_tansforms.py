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

#Resize
tran_resize = transforms.Resize((250,260))
img_resize = tran_resize(img)
img_tensor = trans_tensor(img_resize)


#Compose
t_resize = transforms.Resize(250)
img_t_compose = transforms.Compose([t_resize,trans_tensor])
com_img = img_t_compose(img)
writer.add_image('image_Compose_one', com_img, 0)

#RandomCrop1
for i in range(10):
    t_randcrop = transforms.RandomCrop(100)
    img_rc = t_randcrop(img_tensor)
    writer.add_image("RandomCrop",img_rc,i)

#RandomCrop2
trans_random = transforms.RandomCrop(50)
trans_compe = transforms.Compose([trans_random,trans_tensor])
for j in range(10):
    img_random = trans_compe(img)
    writer.add_image("Random2",img_random,j)

writer.close()