import os

# 文件夹A和文件夹B的路径
dir_A = '/root/DDD/heren/image_roi_all'
dir_B = '/root/DDD/heren/label_all'

# 获取文件夹A和文件夹B中的所有文件
files_A = set(os.listdir(dir_A))
files_B = set(os.listdir(dir_B))

# 找到在文件夹A中有但是在文件夹B中没有的文件
diff_A = files_A - files_B

# 找到在文件夹B中有但是在文件夹A中没有的文件
diff_B = files_B - files_A

# 输出不同名称的文件
if diff_A:
    print(f'文件夹A中有但是文件夹B中没有的文件：{diff_A}')
if diff_B:
    print(f'文件夹B中有但是文件夹A中没有的文件：{diff_B}')
