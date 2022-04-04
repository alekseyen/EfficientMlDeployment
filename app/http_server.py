from fastapi import FastAPI
from pydantic import BaseModel
import logging
import sys
import uvicorn
import grpc
import inference_pb2_grpc
import inference_pb2
import os
from prometheus_client import multiprocess
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST, Counter
from starlette.requests import Request
from starlette.responses import Response


class ImageRequest(BaseModel):
    url: str


class ObjectResponse(BaseModel):
    objects: list[str]


os.environ["PROMETHEUS_MULTIPROC_DIR"] = 'metrics'
app = FastAPI()
registry = CollectorRegistry()

MY_COUNTER = Counter('app_http_inference_count', 'The number of http endpoint invocations.', registry=registry)


@app.post("/predict", response_model=ObjectResponse)
async def predict_endpoint(req: ImageRequest):
    logging.info(f'coming url is {req.url}')

    MY_COUNTER.inc()

    async with grpc.aio.insecure_channel('0.0.0.0:9090') as channel:
        service = inference_pb2_grpc.InstanceDetectorStub(channel)
        r = await service.Predict(inference_pb2.InstanceDetectorInput(
            url=req.url,
        ))

    return ObjectResponse(objects=list(r.objects))


@app.get('/metrics')
def metrics(request: Request):
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    logging.debug(data)

    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    print("http start serving ...")
    uvicorn.run("http_server:app", port=8080, host='0.0.0.0')
