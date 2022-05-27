import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter, writer
#数据加载
CIF_data_train = torchvision.datasets.CIFAR10(root='./data',train=True,transform=transforms.ToTensor(),download=True)
CIF_data_test = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transforms.ToTensor(),download=True)
#tensoboard
Writer = SummaryWriter("logs")
for i in range(10):
    img,label = CIF_data_train[i]
    Writer.add_image('CIF_image', img, i)
    print(img.shape)
    print(label)

Writer.close()
#print("image_shape,label",image.shape,label)
#print(type(CIF_data_train))
#print(CIF_data_train)