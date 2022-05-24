from torch.utils.data import Dataset
from PIL import Image
import os

class OurData(Dataset):
    def __init__(self,root_dir,label_dir) -> None: #初始化根目录和标签路径
        super().__init__()
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_list = os.listdir(self.path)

    def __getitem__(self, index): #获得每一个图片及其标签
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    
    def __len__(self): #数据集有多长
        return len(self.img_list)

root_dir = "hymenoptera_data\\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = OurData(root_dir,ants_label_dir)
bees_dataset = OurData(root_dir,bees_label_dir)

train_dataset = ants_dataset + bees_dataset

