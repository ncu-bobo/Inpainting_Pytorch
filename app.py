import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.app_utils import Inpainter

# 创建inpainter实例并加载模型
inpainter = Inpainter()
inpainter.load_models('app/models.yaml')

# 创建FastAPI应用
app = FastAPI()

# 将文件目录挂载到'/files'和'/static'下
app.mount('/files', StaticFiles(directory='app/files'), 'files')
app.mount('/static', StaticFiles(directory='app/frontend/build/static'), 'static')

# 首页路由
@app.get('/')
async def get_page():
    return FileResponse('app/frontend/build/index.html')

# 获取指定名称的静态文件
@app.get('/{filename}')
async def get_page(filename: str):
    return FileResponse(f'app/frontend/build/{filename}')

# 获取模型信息API路由
@app.get('/api/models')
async def get_model_info():
    model_data = inpainter.get_model_info()
    return {'data': model_data}

# 图像修复API路由
@app.post('/api/inpaint')
async def inpaint(image: UploadFile,
                  mask: UploadFile,
                  models: str = Form(),
                  size: int = Form()):
    response_data = inpainter.inpaint(image, mask, models, size)
    return {'data': response_data}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
