import glob
import random
import shutil

from tqdm import tqdm
from pathlib import Path
from skimage import io

background_images_names = glob.glob('val2017/*')

# remove one-channel images
for name in tqdm(background_images_names):
    img = io.imread(name)
    if len(img.shape) != 3:
        background_images_names.remove(name)
        
random.Random(0).shuffle(background_images_names)

train_background_images = background_images_names[:3000]
val_background_images = background_images_names[3000:4000]
test_background_images = background_images_names[4000:]

Path('./background/train').mkdir(parents=True, exist_ok=True)
Path('./background/val').mkdir(parents=True, exist_ok=True)
Path('./background/test').mkdir(parents=True, exist_ok=True)

for image in tqdm(train_background_images):
    shutil.copy(image, './background/train/' + image[-16:])
    
for image in tqdm(val_background_images):
    shutil.copy(image, './background/val/' + image[-16:])
    
for image in tqdm(test_background_images):
    shutil.copy(image, './background/test/' + image[-16:])

