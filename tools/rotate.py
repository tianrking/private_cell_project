# 引入库
import os
from PIL import Image

# 定义函数
def batch_process_images(image_dir, output_dir, angle=0, flip=False, scale=1.0):
    # 判断输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(image_dir):
        # 判断是否是图片文件
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # 拼接输入文件的完整路径
            input_path = os.path.join(image_dir, file_name)
            # 拼接输出文件的完整路径
            output_path = os.path.join(output_dir, file_name)

            # 打开图片文件
            with Image.open(input_path) as image:
                # 对图片进行旋转
                if angle != 0:
                    image = image.rotate(angle)
                # 对图片进行镜像翻转
                if flip:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 对图片进行缩放
                if scale != 1.0:
                    size = tuple(int(d * scale) for d in image.size)
                    image = image.resize(size, Image.BILINEAR)

                # 保存处理后的图片到输出目录
                new_file_name = f"{file_name}_{angle}_{int(flip)}_{int(scale * 100)}%.jpg"
                image.save(output_path, format='JPEG')

batch_process_images("/content/IMAGE_DATA/DDD/heren-yuanhe/test/image_dir", '/content/IMAGE_DATA/DDD/heren-yuanhe/test/image_dir_90', angle=90, flip=True, scale=1)
