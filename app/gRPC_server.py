import logging
from concurrent import futures
import requests
import grpc
import inference_pb2
import inference_pb2_grpc
import segmentation
import sys


class InferenceDetector(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        self.model = segmentation.model
        self.get_label_picture = segmentation.get_labels_from_picture
        self.transform = segmentation.transform

    def Predict(self, request, context):
        logging.info(f'comming request is: {request.url}')

        with requests.get(request.url) as resp:
            img = resp.content

        image_data = segmentation.transform(img)

        objects = segmentation.get_labels_from_picture(self.model, image_data)

        return inference_pb2.InstanceDetectorOutput(objects=objects)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InferenceDetector(), server)
    server.add_insecure_port('0.0.0.0:9090')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print("start serving...")
    serve()
