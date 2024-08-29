import os
import shutil

# 源目录路径
source_dir = r'/mnt/zetadata/yangyujie/workspace/depression/data/Daic_Woz/no-chunked/text'
# 目标目录路径
target_dir = r'/mnt/zetadata/yangyujie/workspace/depression/data/Daic_Woz/data'

# 遍历源目录中的所有文件
for filename in os.listdir(source_dir):
    # 构造完整的文件路径
    source_file_path = os.path.join(source_dir, filename)

    # 确保是文件而不是目录
    if os.path.isfile(source_file_path):
        # 根据文件名创建目标目录中的子目录
        target_subdir = os.path.join(target_dir, os.path.splitext(filename)[0], "text")
        print(target_subdir)

        # 如果目标子目录不存在，则创建它
        if not os.path.exists(target_subdir):
            os.makedirs(target_subdir)

        # 构造目标文件的完整路径
        target_file_path = os.path.join(target_subdir, filename)

        # 复制文件
        shutil.copy2(source_file_path, target_file_path)
        print(f'Copied {source_file_path} to {target_file_path}')

print('All files have been copied.')