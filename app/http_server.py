from fastapi import FastAPI
from pydantic import BaseModel
import logging
import sys
import uvicorn
import grpc
import inference_pb2_grpc
import inference_pb2

class ImageRequest(BaseModel):
    url: str


class ObjectResponse(BaseModel):
    objects: list[str]


app = FastAPI()


@app.post("/predict", response_model=ObjectResponse)
async def predict_endpoint(req: ImageRequest):

    logging.info(f'comming url is {type(req.url)}')

    async with grpc.aio.insecure_channel('0.0.0.0:9090') as channel:
        service = inference_pb2_grpc.InstanceDetectorStub(channel)
        r = await service.Predict(inference_pb2.InstanceDetectorInput(
            url=req.url,
        ))

    # logging.info(f"want to answer with {r.objects}")

    response_img_objects = r.objects
    # logging.info(f"type is {response_img_objects}")
    # logging.info(f"type is {type(list(response_img_objects))}")

    return ObjectResponse(objects=list(r.objects))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print("http start serving ...")
    uvicorn.run("http_server:app", port=8080, host='0.0.0.0')
    # uvicorn.run("http_server:app", port=9999, host='localhost') # local run
