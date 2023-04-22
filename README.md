## 预训练模型
模型参数: [Places2](https://drive.google.com/u/0/uc?id=1tvdQRmkphJK7FYveNAKSMWC6K09hJoyt&export=download)


## 测试模型
测试之前需要确保已经下载好了模型参数
```bash
python test.py --image examples/inpaint/case1.png --mask examples/inpaint/case1_mask.png --out examples/inpaint/case1_out_test.png --checkpoint pretrained/states_tf_places2.pth
```


## 训练模型
```bash
python train.py --config configs/train.yaml
```
查看Tensorboard日志记录
```bash
tensorboard --logdir <your_log_dir>
```

  
## 构建web app应用
安装fast-api模块

```bash
pip install fastapi python-multipart "uvicorn[standard]"
```
 
Run with:
 `python app.py`
 
New models can be added in `app/models.yaml`


## Requirements
  + python3
  + pytorch
  + torchvision
  + numpy
  + Pillow
  + tensorboard
  + pyyaml
