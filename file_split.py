import os
import shutil
from math import ceil
import random

def split_images(root_folder, train_ratio=0.8):
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        if os.path.isdir(subdir_path):
            # Create train and val directories inside the subdirectory
            train_dir = os.path.join(subdir_path, 'train')
            val_dir = os.path.join(subdir_path, 'val')

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Get a list of image files in the subdirectory
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            random.shuffle(images)
            train_count = int(ceil(len(images) * train_ratio))

            # Move images to train and val directories
            for i, image in enumerate(images):
                src_path = os.path.join(subdir_path, image)

                if i < train_count:
                    dst_path = os.path.join(train_dir, image)
                else:
                    dst_path = os.path.join(val_dir, image)

                shutil.move(src_path, dst_path)

if __name__ == '__main__':
    root_folder = 'D:\Desktop\\asl_alphabet_train\\'
    split_images(root_folder)
