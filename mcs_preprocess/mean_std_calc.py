import numpy as np
from PIL import Image
import glob 


data_dir = '/home/kalyanav/MS_thesis/RPIN_MCS/data/MCS_SS/train' 
image_list = []

for img_path in glob.glob(data_dir + "/****/*.png", recursive=True):
    img = np.array(Image.open(img_path))
    image_list.append(img)
    
images = np.stack(image_list, axis=0)
mean = np.mean(images, axis=(0, 1, 2))
std = np.std(images, axis=(0, 1, 2))

print("Mean: ", mean)
print("Std: ", std)