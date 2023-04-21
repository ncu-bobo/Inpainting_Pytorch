import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# 用于将字典转换为对象
class DictConfig(object):

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

# 从一个配置文件中读取配置信息
def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config

# 将输入的 PyTorch 张量转换为图像格式
def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)

# 用于保存模型的状态字典
# 将生成器、判别器、生成器优化器、判别器优化器以及当前迭代数保存到名称为 fname 的文件中。
def save_states(fname, gen, dis, g_optimizer, d_optimizer, n_iter, config):
    state_dicts = {'G': gen.state_dict(),
                   'D': dis.state_dict(),
                   'G_optim': g_optimizer.state_dict(),
                   'D_optim': d_optimizer.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")

# 将神经网络的输出转换为图像格式
def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out

# 使用 Torch 的推断模式进行计算
@torch.inference_mode()
def infer_deepfill(generator,
                   image,
                   mask,
                   return_vals=['inpainted', 'stage1']):

    _, h, w = image.shape
    # 将图像划分为大小为 8x8 的栅格，以提高计算效率
    grid = 8
    # 裁剪图像和 mask，以使它们的高度和宽度都是 8 的倍数
    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    # 将图像像素值映射到 [-1, 1] 范围内
    image = (image*2 - 1.)
    # 1.: masked  0.: unmasked
    mask = (mask > 0.).to(dtype=torch.float32)

    # 对输入图像进行遮罩处理
    image_masked = image * (1.-mask)  # mask image

    # 生成 inpainting 结果的阶段 1 和阶段 2
    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)

    x_stage1, x_stage2 = generator(x, mask)

    # 对 inpainting 结果进行补全
    image_compl = image * (1.-mask) + x_stage2 * mask

    # 存储返回的结果列表
    output = []
    for return_val in return_vals:
        if return_val.lower() == 'stage1':
            output.append(output_to_img(x_stage1))
        elif return_val.lower() == 'stage2':
            output.append(output_to_img(x_stage2))
        elif return_val.lower() == 'inpainted':
            output.append(output_to_img(image_compl))
        else:
            print(f'Invalid return value: {return_val}')

    return output

# 生成随机的 t,l,h,w 参数，以指定输入图像中需要进行修复的区域。
def random_bbox(config):

    img_height, img_width, _ = config.img_shapes
    maxt = img_height - config.vertical_margin - config.height
    maxl = img_width - config.horizontal_margin - config.width
    t = np.random.randint(config.vertical_margin, maxt)
    l = np.random.randint(config.horizontal_margin, maxl)

    return (t, l, config.height, config.width)

# 根据 bbox 生成与之相对应的遮罩张量
def bbox2mask(config, bbox):

    img_height, img_width, _ = config.img_shapes
    mask = torch.zeros((1, 1, img_height, img_width),
                       dtype=torch.float32)
    h = np.random.randint(config.max_delta_height // 2 + 1)
    w = np.random.randint(config.max_delta_width // 2 + 1)
    mask[:, :, bbox[0]+h: bbox[0]+bbox[2]-h,
         bbox[1]+w: bbox[1]+bbox[3]-w] = 1.
    return mask

# 生成画笔笔触遮罩
def brush_stroke_mask(config):
    min_num_vertex = 4
    max_num_vertex = 12
    min_width = 12
    max_width = 40

    mean_angle = 2*np.pi / 5
    angle_range = 2*np.pi / 15

    H, W, _ = config.img_shapes

    # 随机生成 1~3 个笔触
    average_radius = np.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)

    # 随机生成 1~3 个笔触
    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            # 对于奇数和偶数个角分别随机生成对应的夹角
            if i % 2 == 0:
                angles.append(
                    2*np.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)),
                      int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * np.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * np.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))
        # 创建一个绘图对象
        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    # 将 PIL 图像对象转换为 Numpy 数组
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 1, H, W))
    # 返回画笔笔触的张量
    return torch.Tensor(mask)

# 测试上下文注意力层
def test_contextual_attention(imageA, imageB, contextual_attention):
    rate = 2
    stride = 1
    grid = rate*stride

    # 读取图像A
    b = Image.open(imageA)
    # 缩小图像 A 的尺寸，以便进行处理
    b = b.resize((b.width//2, b.height//2), resample=Image.BICUBIC)
    b = T.ToTensor()(b)

    _, h, w = b.shape
    # 调整图像 A 的尺寸，以确保能被整除
    b = b[:, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print('Size of imageA: {}'.format(b.shape))

    # 读取图像b
    f = T.ToTensor()(Image.open(imageB))
    _, h, w = f.shape
    f = f[:, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print('Size of imageB: {}'.format(f.shape))

    yt, flow = contextual_attention(f*255., b*255.)

    return yt, flow
