import os
import shutil


root_dir = '/home/zy/wjj/dataset/QNRF'
images_dir = os.path.join(root_dir, 'images')
ground_truth_dir = os.path.join(root_dir, 'ground_truth')
labels_dir = os.path.join(root_dir, 'labels')
train_txt = os.path.join(root_dir, 'train.txt')
val_txt = os.path.join(root_dir, 'val.txt')
test_txt = os.path.join(root_dir, 'test.txt')



def read_file_list(txt_path):
    with open(txt_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


train_files = read_file_list(train_txt)
val_files = read_file_list(val_txt)
test_files = read_file_list(test_txt)


def create_directory(dataset_type, file_type):
    target_dir = os.path.join(root_dir, dataset_type, file_type)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir



def move_files(file_list, dataset_type):

    move_image_dir = create_directory(dataset_type, 'images')

    move_gt_dir = create_directory(dataset_type, 'ground_truth')

    move_label_dir = create_directory(dataset_type, 'labels')

    for file_name in file_list:
        image_path = os.path.join(images_dir, f"{file_name}.jpg")
        gt_path = os.path.join(ground_truth_dir, f"{file_name}.mat")
        label_path = os.path.join(labels_dir, f"{file_name}.png")

        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(move_image_dir, f"{file_name}.jpg"))
        if os.path.exists(gt_path):
            shutil.move(gt_path, os.path.join(move_gt_dir, f"{file_name}.mat"))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(move_label_dir,  f"{file_name}.png"))


move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')
