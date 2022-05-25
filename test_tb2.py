import torch
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
img_path = "hymenoptera_data\\train\\ants\\5650366_e22b7e1065.jpg"
img = Image.open(img_path)
img_array = np.array(img)
writer.add_image("train", img_array, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar("2*x = y",i,2*i)
writer.close()