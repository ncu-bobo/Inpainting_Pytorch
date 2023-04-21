import torch

# 加载生成器模型
def load_model(path, device='cpu'):
    try:  
        gen_sd = torch.load(path, map_location=device)['G']
    except FileNotFoundError:
        return None

    if 'stage1.conv1.conv.weight' in gen_sd.keys():
        from model.networks import Generator     
    else:
        from model.networks_tf import Generator 

    gen = Generator(cnum_in=5, cnum=48, return_flow=False)
    gen = gen.to(device)
    gen.eval()

    gen.load_state_dict(gen_sd, strict=False)
    return gen
