import argparse
from PIL import Image
import torch
import torchvision.transforms as T

# 参数解析器
parser = argparse.ArgumentParser(description='Test inpainting')
parser.add_argument("--image", type=str,
                    default="examples/inpaint/case1.png", help="path to the image file")
parser.add_argument("--mask", type=str,
                    default="examples/inpaint/case1_mask.png", help="path to the mask file")
parser.add_argument("--out", type=str,
                    default="examples/inpaint/case1_out.png", help="path for the output file")
parser.add_argument("--checkpoint", type=str,
                    default="pretrained/states_pt_places2.pth", help="path to the checkpoint file")


def main():

    args = parser.parse_args()
    device = torch.device('cpu')
    # 加载生成器模型参数
    generator_state_dict = torch.load(args.checkpoint, map_location=device)['G']
    # 选择pytorch
    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator

    # use_cuda_if_available = True


    # 设置网络架构
    # 5个特征向量，48个卷积核，并使用cpu
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator_state_dict = torch.load(args.checkpoint, map_location=device)['G']
    # 将加载的模型参数赋值给生成器模型
    generator.load_state_dict(generator_state_dict, strict=True)

    # 加载Image和mask
    image = Image.open(args.image)
    mask = Image.open(args.mask)

    # 图像预处理
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    # 获取图像通道数，高度，宽度特征
    _, h, w = image.shape
    grid = 8
    # 选择前三个通道，向下取整并乘以网格大小，确保能被均匀的划分为8个小块
    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    # 只选择前一个通道
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    # 将图像张量的像素值映射到[-1,1]之间的范围
    image = (image*2 - 1.).to(device)
    # 将掩模张量中大于0.5的像素值置为1，小于等于0.5的像素值置为0
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)

    # 根据掩模张量和图像张量的点积结果来生成一个被掩蔽的图像张量
    image_masked = image * (1.-mask)

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    # 拼接后的张量x由三部分组成：image_masked、全1的张量ones_x和经过mask操作后的全1张量ones_x * mask
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1.-mask) + x_stage2 * mask

    # 保存修复后的图片
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    img_out.save(args.out)

    print(f"Saved output file at: {args.out}")


if __name__ == '__main__':
    main()
