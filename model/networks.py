import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------

# 初始化卷积层
def _init_conv_layer(conv, activation, mode='fan_out'):
    if isinstance(activation, nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       a=activation.negative_slope,
                                       nonlinearity='leaky_relu',
                                       mode=mode)
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       nonlinearity='relu',
                                       mode=mode)
    else:
        pass
    if conv.bias != None:
        torch.nn.init.zeros_(conv.bias)

# 将输出转换成图像
def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out

# ----------------------------------------------------------------------------

#################################
########### GENERATOR ###########
#################################

# 生成器卷积网络
class GConv(nn.Module):

    def __init__(self, cnum_in,
                 cnum_out,
                 ksize,
                 stride=1,
                 padding='auto',
                 rate=1,
                 activation=nn.ELU(),
                 bias=True
                 ):

        super().__init__()

        padding = rate*(ksize-1)//2 if padding == 'auto' else padding
        self.activation = activation
        self.cnum_out = cnum_out
        num_conv_out = cnum_out if self.cnum_out == 3 or self.activation is None else 2*cnum_out

        # 定义一个2D卷积层
        self.conv = nn.Conv2d(cnum_in,
                              num_conv_out,
                              kernel_size=ksize,
                              stride=stride,
                              padding=padding,
                              dilation=rate,
                              bias=bias)

        _init_conv_layer(self.conv, activation=self.activation)

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.padding = padding

    # 前向传播方法
    def forward(self, x):
        x = self.conv(x)
        if self.cnum_out == 3 or self.activation is None:
            return x
        x, y = torch.split(x, self.cnum_out, dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

# ----------------------------------------------------------------------------

class GDeConv(nn.Module):

    def __init__(self, cnum_in,
                 cnum_out,
                 padding=1):
        super().__init__()
        self.conv = GConv(cnum_in, cnum_out, 3, 1,
                          padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest',
                          recompute_scale_factor=False)
        x = self.conv(x)
        return x

# ----------------------------------------------------------------------------

# 下采样操作
class GDownsamplingBlock(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out,
                 cnum_hidden=None
                 ):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, 3, 2)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, 1)

    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x

# ----------------------------------------------------------------------------

# 上采样操作
class GUpsamplingBlock(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out,
                 cnum_hidden=None
                 ):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, 1)

    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        return x

# ----------------------------------------------------------------------------


class CoarseGenerator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = GConv(cnum_in, cnum//2, 5, 1, padding=2)

        # 下采样
        self.down_block1 = GDownsamplingBlock(cnum//2, cnum)
        self.down_block2 = GDownsamplingBlock(cnum, 2*cnum)

        self.conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
        self.conv_bn3 = GConv(2*cnum, 2*cnum, 3, rate=4, padding=4)
        self.conv_bn4 = GConv(2*cnum, 2*cnum, 3, rate=8, padding=8)
        self.conv_bn5 = GConv(2*cnum, 2*cnum, 3, rate=16, padding=16)
        self.conv_bn6 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_bn7 = GConv(2*cnum, 2*cnum, 3, 1)

        # 上采样
        self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

        # 转换为RGB
        self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)

        # 下采样
        x = self.down_block1(x)
        x = self.down_block2(x)

        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = self.conv_bn4(x)
        x = self.conv_bn5(x)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)

        # 上采样
        x = self.up_block1(x)
        x = self.up_block2(x)

        # 转换为RGB
        x = self.conv_to_rgb(x)
        x = self.tanh(x)
        return x

# ----------------------------------------------------------------------------

class FineGenerator(nn.Module):
    def __init__(self, cnum, return_flow=False):
        super().__init__()

        # 门控卷积
        self.conv_conv1 = GConv(3, cnum//2, 5, 1, padding=2)

        # 下采样
        self.conv_down_block1 = GDownsamplingBlock(
            cnum//2, cnum, cnum_hidden=cnum//2)
        self.conv_down_block2 = GDownsamplingBlock(
            cnum, 2*cnum, cnum_hidden=cnum)

        # bottleneck
        self.conv_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
        self.conv_conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
        self.conv_conv_bn3 = GConv(2*cnum, 2*cnum, 3, rate=4, padding=4)
        self.conv_conv_bn4 = GConv(2*cnum, 2*cnum, 3, rate=8, padding=8)
        self.conv_conv_bn5 = GConv(2*cnum, 2*cnum, 3, rate=16, padding=16)

        # 注意力分支
        self.ca_conv1 = GConv(3, cnum//2, 5, 1, padding=2)

        # 下采样
        self.ca_down_block1 = GDownsamplingBlock(
            cnum//2, cnum, cnum_hidden=cnum//2)
        self.ca_down_block2 = GDownsamplingBlock(cnum, 2*cnum)

        # bottleneck
        self.ca_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1, activation=nn.ReLU())
        self.contextual_attention = ContextualAttention(ksize=3,
                                                        stride=1,
                                                        rate=2,
                                                        fuse_k=3,
                                                        softmax_scale=10,
                                                        fuse=True,
                                                        device_ids=None,
                                                        return_flow=return_flow,
                                                        n_down=2)
        self.ca_conv_bn4 = GConv(2*cnum, 2*cnum, 3, 1)
        self.ca_conv_bn5 = GConv(2*cnum, 2*cnum, 3, 1)

        # 合并上下文
        self.conv_bn6 = GConv(4*cnum, 2*cnum, 3, 1)
        self.conv_bn7 = GConv(2*cnum, 2*cnum, 3, 1)

        # 上采样
        self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

        # 转换为RGB
        self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
        self.tanh = nn.Tanh()

    # 前向传播
    def forward(self, x, mask):
        xnow = x

        # 卷积分支
        x = self.conv_conv1(xnow)
        # 下采样
        x = self.conv_down_block1(x)
        x = self.conv_down_block2(x)

        # bottleneck
        x = self.conv_conv_bn1(x)
        x = self.conv_conv_bn2(x)
        x = self.conv_conv_bn3(x)
        x = self.conv_conv_bn4(x)
        x = self.conv_conv_bn5(x)
        x_hallu = x

        # 注意力分支
        x = self.ca_conv1(xnow)
        # 下采样
        x = self.ca_down_block1(x)
        x = self.ca_down_block2(x)

        # bottleneck
        x = self.ca_conv_bn1(x)
        x, offset_flow = self.contextual_attention(x, x, mask)
        x = self.ca_conv_bn4(x)
        x = self.ca_conv_bn5(x)
        pm = x

        # 连接两个分支的输出
        x = torch.cat([x_hallu, pm], dim=1)

        # 合并上下文
        x = self.conv_bn6(x)
        x = self.conv_bn7(x)

        # 上采样
        x = self.up_block1(x)
        x = self.up_block2(x)

        # 转换为RGB
        x = self.conv_to_rgb(x)
        x = self.tanh(x)

        return x, offset_flow

# ----------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, cnum_in=5, cnum=48, return_flow=False, checkpoint=None):
        super().__init__()
        # 定义两个生成器
        self.stage1 = CoarseGenerator(cnum_in, cnum)
        self.stage2 = FineGenerator(cnum, return_flow)
        self.return_flow = return_flow

        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint)['G']
            self.load_state_dict(generator_state_dict, strict=True)

        self.eval()

    def forward(self, x, mask):
        xin = x
        # 获取粗略生成器
        x_stage1 = self.stage1(x)
        # 生成粗略修复图像
        x = x_stage1*mask + xin[:, 0:3, :, :]*(1.-mask)
        # 获取细节生成器
        x_stage2, offset_flow = self.stage2(x, mask)

        if self.return_flow:
            return x_stage1, x_stage2, offset_flow

        return x_stage1, x_stage2

    # 推理方法，用于测试图像修复的效果
    @torch.inference_mode()
    def infer(self,
              image,
              mask,
              return_vals=['inpainted', 'stage1'],
              device='cuda'):

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        # 调整图像范围到[-1, 1]之间
        image = (image*2 - 1.)
        # 1.: masked 0.: unmasked
        mask = (mask > 0.).to(dtype=torch.float32)

        image_masked = image * (1.-mask)

        # 生成sketch通道
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        # 拼接通道
        x = torch.cat([image_masked, ones_x, ones_x*mask],
                      dim=1)

        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)

        # 将修复后的结果与原图中不需要修复的区域进行拼接
        image_compl = image * (1.-mask) + x_stage2 * mask

        output = []
        for return_val in return_vals:
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')

        return output

# ----------------------------------------------------------------------------

####################################
####### 上下文注意力层 #######
####################################


class ContextualAttention(nn.Module):

    def __init__(self,
                 ksize=3, # 卷积核大小
                 stride=1, # 步长
                 rate=1, # 扩张系数
                 fuse_k=3,
                 softmax_scale=10.,
                 n_down=2, # 下采样层数
                 fuse=False, # 融合卷积
                 return_flow=False,  # 是否返回上下文注意力中的流
                 device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device_ids = device_ids
        self.n_down = n_down
        self.return_flow = return_flow
        self.register_buffer('fuse_weight', torch.eye(
            fuse_k).view(1, 1, fuse_k, fuse_k))

    # 前向传播
    def forward(self, f, b, mask=None):

        device = f.device
        # 获取形状参数
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())   # b*c*h*w

        # 根据步长和扩张率提取补丁
        kernel = 2 * self.rate
        raw_w = extract_image_patches(b, ksize=kernel,
                                      stride=self.rate*self.stride,
                                      rate=1, padding='auto')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # 下采样整张图片
        f = F.interpolate(f, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        b = F.interpolate(b, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        int_fs, int_bs = list(f.size()), list(b.size())   # b*c*h*w
        # 切割 tensors
        f_groups = torch.split(f, 1, dim=0)
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        # w shape: [N, L, C, k, k]
        w = w.permute(0, 4, 1, 2, 3)
        w_groups = torch.split(w, 1, dim=0)

        # 处理mask
        if mask is None:
            mask = torch.zeros(
                [int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            # 控制下采样率
            mask = F.interpolate(
                mask, scale_factor=1./((2**self.n_down)*self.rate), mode='nearest', recompute_scale_factor=False)
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]

        mm = (torch.mean(m, dim=[1, 2, 3], keepdim=True) == 0.).to(
            torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        scale = self.softmax_scale

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            # 比较并进行卷积运算
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(torch.sum(torch.square(wi), dim=[
                                1, 2, 3], keepdim=True)).clamp_min(1e-4)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            yi = F.conv2d(xi, wi_normed, stride=1, padding=(
                self.ksize-1)//2)   # [1, L, H, W]
            # 激励更大的patches
            if self.fuse:
                # (B=1, I=1, H=32*32, W=32*32)
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                # (B=1, C=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                # (B=1, 32, 32, 32, 32)
                yi = yi.contiguous().view(
                    1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)

                yi = yi.contiguous().view(
                    1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                yi = yi.contiguous().view(
                    1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            # (B=1, C=32*32, H=32, W=32)
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            # softmax
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            if self.return_flow:
                offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

                if int_bs != int_fs:
                    # 归一化
                    times = (int_fs[2]*int_fs[3])/(int_bs[2]*int_bs[3])
                    offset = ((offset + 1).float() * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'),
                                    offset % int_fs[3]], dim=1)  # 1*2*H*W
                offsets.append(offset)

            # 反卷积实现补丁拼接
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(
                yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)
        y = y.contiguous().view(raw_int_fs)

        if not self.return_flow:
            return y, None

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        h_add = torch.arange(int_fs[2], device=device).view(
            [1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view(
            [1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        # 流化
        flow = torch.from_numpy(flow_to_image(
            offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate,
                                 mode='bilinear', align_corners=True)

        return y, flow

# ----------------------------------------------------------------------------

# 将流转换为图像
def flow_to_image(flow):
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

# ----------------------------------------------------------------------------

# 计算color
def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img

# ----------------------------------------------------------------------------

# 生成颜色查找表
def make_color_wheel():
    # 颜色情况的六种情况
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    # 颜色信道数
    ncols = RY + YG + GC + CB + BM + MR
    # 初始化颜色矩阵
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    # 红黄渐变，从红到黄
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    # 黄绿渐变，从黄到绿
    colorwheel[col:col + YG, 0] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    # 绿青渐变，从绿到青
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    # 青蓝渐变，从青到蓝
    colorwheel[col:col + CB, 1] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    # 蓝品红渐变，从蓝到品红
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    # 品红红渐变，从品红到红
    colorwheel[col:col + MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    # 返回颜色矩阵
    return colorwheel


# ----------------------------------------------------------------------------

# 将输入的图像切分成固定尺寸的块
def extract_image_patches(images, ksize, stride, rate, padding='auto'):
    # 自动计算padding大小
    padding = rate*(ksize-1)//2 if padding == 'auto' else padding

    # 创建torch.nn.Unfold对象
    unfold = torch.nn.Unfold(kernel_size=ksize,
                             dilation=rate,
                             padding=padding,
                             stride=stride)
    # 对输入图片进行裁剪，返回裁剪后的图像块
    patches = unfold(images)
    return patches

# ----------------------------------------------------------------------------

#################################
######### DISCRIMINATOR #########
#################################
# 谱归一化卷积层
class Conv2DSpectralNorm(nn.Conv2d):

    def __init__(self, cnum_in,
                 cnum_out, kernel_size, stride, padding=0, n_iter=1, eps=1e-12, bias=True):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    # 正则化
    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):
        # 将权重矩阵展平并分离
        weight_orig = self.weight.flatten(1).detach()
        # 迭代n_iter次求解v和u
        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)
        # 计算sigma值并对权重矩阵进行归一化
        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)
        # 在卷积层的基础上调用父类forward()方法进行前向传播
        x = super().forward(x)

        return x

# ----------------------------------------------------------------------------
# 鉴别器卷积层
class DConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out, ksize=5, stride=2, padding='auto'):
        super().__init__()
        padding = (ksize-1)//2 if padding == 'auto' else padding
        # 谱归一化
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)

        # 使用LeakyRelu激活函数
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x

# 鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, cnum_in, cnum):
        super().__init__()
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)
        self.conv6 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.Flatten()(x)

        return x

