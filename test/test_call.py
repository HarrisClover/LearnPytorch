from torchvision import transforms
from PIL import Image
class test_call:
    def __init__(self,num) -> None:
        self.num = num

    def __call__(self, img):
        self.img = img
        tran = transforms.ToTensor()
        return tran(self.img),self.num

imgtran = test_call(1)
img_dir = "D:\\CodeAllYouNeed\\learnpytorch\\hymenoptera_data\\train\\bees\\36900412_92b81831ad.jpg"
img = Image.open(img_dir)
print(imgtran(img))