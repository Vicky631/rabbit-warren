file_path = "/home/zy/wjj/Prompt_sam_localization/dataset/NWPU/train_v0.txt"  # 将此处替换为实际的txt文件路径
new_lines = []
with open(file_path, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split()
        new_lines.append(parts[0] + '\n')

new_file_path = "/home/zy/wjj/Prompt_sam_localization/dataset/NWPU/train.txt"
with open(new_file_path, 'w') as f:
    f.writelines(new_lines)