import os
import shutil
import time

def save_first_last_files(folder_path, output_folder):
    # 检查目标文件夹是否存在
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"target '{folder_path}' ")
        return

    # 创建保存第一个文件和最后一个文件的文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取目标文件夹中文件夹的列表
    subfolders = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # 保存每个文件夹里的第一个文件和最后一个文件的文件名及路径
    for subfolder in subfolders:
        files = [file for file in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, file))]

        if len(files) == 5:
            #print(files)
            # 创建保存第一个文件和最后一个文件的文件夹
            first_file_folder = os.path.join(output_folder, "first_files")
            last_file_folder = os.path.join(output_folder, "dap_files")

            if not os.path.exists(first_file_folder):
                os.makedirs(first_file_folder)
            if not os.path.exists(last_file_folder):
                os.makedirs(last_file_folder)

            

            

            # 拷贝第一个文件和最后一个文件到对应的文件夹中
            first_file_path = os.path.join(subfolder, files[0])
            last_file_path = os.path.join(subfolder, files[-1])
        
            shutil.copy(first_file_path, os.path.join(first_file_folder, os.path.basename(files[0])))
            shutil.copy(last_file_path, os.path.join(last_file_folder, os.path.basename(files[-1])))
            
    #print(f"save '{output_folder}'.")



def detect_new_files(target_folder, last_seen_files):
    # 检查目标文件夹是否存在
    if not os.path.exists(target_folder) or not os.path.isdir(target_folder):
        print(f"目标文件夹 '{target_folder}' 不存在或不是一个文件夹。")
        return []

    # 获取目标文件夹中所有文件的路径
    current_files = []
    for root, dirs, files in os.walk(target_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            current_files.append(file_path)

    # 比较当前文件和上次检测的文件，找出新增的文件
    new_files = [file for file in current_files if file not in last_seen_files]

    return new_files
