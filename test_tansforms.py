from torchvision import transforms
from PIL import Image
#import cv2
from torch.utils.tensorboard import SummaryWriter

img_dir = "hymenoptera_data\\train\\ants\\0013035.jpg"
img = Image.open(img_dir)
#tensor_img = cv2.()
writer = SummaryWriter("logs")
tran = transforms.ToTensor()
tensor_img = tran(img)
writer.add_image('my_image_HWC', tensor_img)
writer.close()