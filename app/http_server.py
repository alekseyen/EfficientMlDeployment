import asyncio
import concurrent.futures
import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import segmentation
import sys
import uvicorn
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
model = segmentation.model


class ImageRequest(BaseModel):
    url: str


class LabelResponse(BaseModel):
    objects: list[str]


app = FastAPI()


@app.post("/predict", response_model=LabelResponse)
async def predict_endpoint(req: ImageRequest):
    async with aiohttp.ClientSession() as session:
        async with session.get(req.url) as resp:
            data = await resp.read()
            loop = asyncio.get_event_loop()
            image_data = await loop.run_in_executor(executor, segmentation.transform, data)

    labels = list(segmentation.get_labels_from_picture(model, image_data))
    logging.info(f'find next objects {labels}')
    return LabelResponse(objects=labels)


if __name__ == '__main__':
    os.environ['DOCKER_IP'] = "localhost"  # todo: remove, just for debugging
    import json

    with open('eval.json', 'r') as f:
        ttt = json.loads(f.read())
        for i in ttt:
            print(i, end='\t')

    uvicorn.run("http_server:app", port=8080, host=os.environ['DOCKER_IP'])  # todo: поменять, иначе не соберётся
