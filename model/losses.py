import torch

# 采用平方误差作为 GAN 的损失函数
# 定义鉴别器的损失函数
def ls_loss_d(pos, neg, value=1.):
    l2_pos = torch.mean((pos-value)**2)
    l2_neg = torch.mean(neg**2)
    d_loss = 0.5*l2_pos + 0.5*l2_neg 
    return d_loss

# 定义生成器的损失函数
def ls_loss_g(neg, value=1.):    
    """
    gan with least-square loss
    """
    g_loss = torch.mean((neg-value)**2)
    return g_loss


# 通过使用 max 函数作为GAN的损失函数
# 定义鉴别器的损失函数
def hinge_loss_d(pos, neg):
    hinge_pos = torch.mean(torch.relu(1-pos))
    hinge_neg = torch.mean(torch.relu(1+neg))
    d_loss = 0.5*hinge_pos + 0.5*hinge_neg   
    return d_loss

# 定义生成器的损失函数
def hinge_loss_g(neg):
    g_loss = -torch.mean(neg)
    return g_loss