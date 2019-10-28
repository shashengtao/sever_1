import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.responses import JSONResponse
import time
import os
export_file_url = 'https://www.kaggleusercontent.com/kf/22073878/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Uq5Wr5Ie4BLV3DEuWWmRHw.5Pdd8yvRde7cuddlTLMv4mAT7bcKORiwBSKWyAZ48BFw453A6CaXKrIsuL0eC2ZC3ev2iLHyIqw7l-65qmOzqbYVsZrLpNPRI9C-gOseGp4YXPP9-Et-J8YYfREKYTC37JKKkqugmCamo2syCy-RFyr2pGmiInJSkNDC8h46uzw.oIPh3wNDiDydfa8iUxXsLw/export.pkl'

export_file_name = 'export.pkl'

destination = './'

classes={
        "一次性快餐盒":"其他垃圾",
        "污损塑料":"其他垃圾",
        "烟蒂":"其他垃圾",
        "牙签":"其他垃圾",
        "破碎花盆及碟碗":"其他垃圾",
        "竹筷":"其他垃圾",
        "剩饭剩菜":"厨余垃圾",
        "大骨头":"厨余垃圾",
        "水果果皮":"厨余垃圾",
        "水果果肉":"厨余垃圾",
        "茶叶渣":"厨余垃圾",
        "菜叶菜根":"厨余垃圾",
        "蛋壳":"厨余垃圾",
        "鱼骨":"厨余垃圾",
        "充电宝":"可回收物",
        "包":"可回收物",
        "化妆品瓶":"可回收物",
        "塑料玩具":"可回收物",
        "塑料碗盆":"可回收物",
        "塑料衣架":"可回收物",
        "快递纸袋":"可回收物",
        "插头电线":"可回收物",
        "旧衣服":"可回收物",
        "易拉罐":"可回收物",
        "枕头":"可回收物",
        "毛绒玩具":"可回收物",
        "洗发水瓶":"可回收物",
        "玻璃杯":"可回收物",
        "皮鞋":"可回收物",
        "砧板":"可回收物",
        "纸板箱":"可回收物",
        "调料瓶":"可回收物",
        "酒瓶":"可回收物",
        "金属食品罐":"可回收物",
        "锅":"可回收物",
        "可食用油桶":"可回收物",
        "饮料瓶":"可回收物",
        "干电池":"有害垃圾",
        "软膏":"有害垃圾",
        "过期药物":"有害垃圾"
    }
path = Path(__file__).parent
app = Starlette()

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/search', methods=['POST'])
async def search(request):
    img_data = await request.form()
    img_bytes = await (img_data['image'].read())
    img = open_image(BytesIO(img_bytes))
    output = learn.predict(img)[0]
    prediction = output.obj + "-" + classes[output.obj]
    return JSONResponse({"result": prediction})

@app.route('/upload', methods=['POST'])
async def upload(request):
    img_data = await request.form()
    img_bytes = await (img_data['image'].read())
    img_type = img_data['class']
    img = open_image(BytesIO(img_bytes))
    destination = os.path.join("./", img_type)
    if os.path.exists(destination) is False:
        os.mkdir(destination)
    total = len(os.listdir(destination))
    img.save(os.path.join(destination, str(total)+".jpg"))
    return JSONResponse()

if __name__ == '__main__':
    uvicorn.run(app=app, host='119.3.222.50', port=5000, log_level="info")
