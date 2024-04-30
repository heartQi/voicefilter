import os

# 定义要重命名的文件夹路径
folder_path = './generator2/train'

# 获取文件夹中所有的文件名
files = os.listdir(folder_path)

# 循环处理每个文件
for file in files:
    # 构建新的文件名，前面添加'local-'
    new_name = 'local-' + file
    # 构建原始文件路径和新文件路径
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    # 重命名文件
    os.rename(old_path, new_path)
    print(f'Renamed {file} to {new_name}')
