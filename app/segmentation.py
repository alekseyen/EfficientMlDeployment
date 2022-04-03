import io

import torchvision
import torch
from PIL import Image
import torchvision.transforms as transforms

# from torchvision.models.detection.transform import GeneralizedRCNNTransform

COCO_INSTANCE_CATEGORY_NAMES = ([
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])

device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=False).to(device).eval()


def get_labels_from_picture(model: 'torchvision.models', img, detection_score=0.75):
    prediction = model(img)[0]
    objects = []

    for label, score in zip(prediction['labels'], prediction['scores']):
        if score >= detection_score:
            objects.append(COCO_INSTANCE_CATEGORY_NAMES[label])

    if len(objects) == 0:
        return ['__background__']

    return objects


def transform(img_data):
    image = Image.open(io.BytesIO(img_data))
    # image_mean = [0.485, 0.456, 0.406]
    # image_std = [0.229, 0.224, 0.225]
    # min_size = 800
    # max_size = 1333
    # return GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)(img)

    return transforms.Compose([transforms.ToTensor(), ])(image).unsqueeze(0)


__all__ = [get_labels_from_picture, transform]

## testing
# if __name__ == '__main__':
#     from PIL import Image
#     import torchvision
#     import torchvision.transforms as transforms
#     from PIL import Image
#
#     transform_pipeline = transforms.Compose([
#         # transforms.Resize(224),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     image = Image.open("../img/milk_vase_book_clock.jpg")
#     img_data = transform(image)
#
#     print(model(img_data)[0]['labels'])
