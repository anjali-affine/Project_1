import glob
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import imgaug.augmenters as iaa

def augment_it(img):
    seq = iaa.Sometimes(0.1, [
    iaa.CoarseDropout((0,0.02), size_percent=(0.10,0.4)),
    ])
    aug_img = seq(image=img)
    return aug_img


datagen = ImageDataGenerator(preprocessing_function=augment_it, fill_mode="wrap")


for f in glob.glob("data_11-whitespace-aug/val/**/*g", recursive=True):
    print(f)
    head, tail = os.path.split(f)
    img = load_img(f)  
    x = img_to_array(img) 
    # Reshape the input image 
    x = x.reshape((1, ) + x.shape)  
    i = 0

    # generate 5 new augmented images 
    for batch in datagen.flow(x, batch_size = 1, 
                      save_to_dir = head, save_prefix ='aug1'):
        i += 1
        if i > 1: 
            break

