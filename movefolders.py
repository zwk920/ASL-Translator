import os
import shutil

def combine_folders(root_folder):
    main_train_dir = os.path.join(root_folder, 'train')
    main_val_dir = os.path.join(root_folder, 'val')

    os.makedirs(main_train_dir, exist_ok=True)
    os.makedirs(main_val_dir, exist_ok=True)

    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        if os.path.isdir(subdir_path) and subdir not in ['train', 'val']:
            train_dir = os.path.join(subdir_path, 'train')
            val_dir = os.path.join(subdir_path, 'val')

            if os.path.exists(train_dir):
                for image in os.listdir(train_dir):
                    src_path = os.path.join(train_dir, image)
                    dst_path = os.path.join(main_train_dir, f"{subdir}_{image}")
                    shutil.move(src_path, dst_path)

            if os.path.exists(val_dir):
                for image in os.listdir(val_dir):
                    src_path = os.path.join(val_dir, image)
                    dst_path = os.path.join(main_val_dir, f"{subdir}_{image}")
                    shutil.move(src_path, dst_path)

if __name__ == '__main__':
    root_folder = 'D:\Desktop\\asl_alphabet_train\\'
    combine_folders(root_folder)
