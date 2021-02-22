import json
import random
import time
import os
import io
import boto3
import json
import csv
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import sagemaker
from sagemaker import get_execution_role
import botocore
import uuid

AWS_DEFAULT_REGION = 'us-east-1'
role = get_execution_role()
print(role)
sess = sagemaker.Session()
s3 = boto3.client('s3')

output_bucket = 'sro-ab3-inference-output'


def get_image(bucket,key):
    newpath = '/tmp/{}.jpg'.format(str(uuid.uuid1()))
    s3.download_file(bucket, key, newpath)
    print('get image done')
    return newpath

def visualize_detection(newpath, dets, classes=[], thresh=0.4):
    img = mpimg.imread(newpath)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for det in dets:
        (klass, score, x0, y0, x1, y1) = det
        if score < thresh:
            continue
        cls_id = int(klass)
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=1)
        plt.gca().add_patch(rect)
        class_name = str(cls_id)
        if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
        plt.gca().text(xmin, ymin - 2,
                       '{:s} {:.3f}'.format(class_name, score),
                       bbox=dict(facecolor=colors[cls_id], alpha=0.6),
                       fontsize=6, color='white')

    plt.savefig('/tmp/predictedimage.png')
    plt.show()
    print("Success... prepare to upload")
    bucket = 'sro-ab3-inference-output'
    output_prefix = 'defective'
    unique_id2 = str(uuid.uuid4())
    s3_predicted_location = '{}/{}-od_defects.png'.format(output_prefix, unique_id2)
    s3.upload_file(newpath, output_bucket, s3_predicted_location)


def second_infer(bucket,key):
    print("Starting second inference")
    newpath = get_image(bucket, key)
    od_predictor = sagemaker.predictor.Predictor("ab3--od-inference")
    with open(newpath, 'rb') as image:
        f = image.read()
        b = bytearray(f)
        ne = open('/tmp/n.txt', 'wb')
        ne.write(b)
        #os.remove('/tmp/n.txt')
    results = od_predictor.predict(b, initial_args={'ContentType': 'image/jpeg'})
    detections = json.loads(results)
    print(detections)
    object_categories = ['holes', 'resin', 'malformed']

    # Setting a threshold 0.20 will only plot detection results that have a confidence score greater than 0.20.
    threshold = 0.2

    # Visualize the detections.
    visualize_detection(newpath, detections['prediction'], object_categories, threshold)


def run_acceptable(newpath):
    output_prefix = 'acceptable'
    #upload_resized = '/tmp/newpath.jpg'
    # background = Image.open(newpath)
    # background.save('/tmp/highlight_mask.png', "PNG")
    unique_id = str(uuid.uuid1())
    s3_predicted_location = '{}/{}-no_defects.png'.format(output_prefix, unique_id)
    s3.upload_file(newpath, output_bucket, s3_predicted_location)


def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        print(bucket)
        key = record['s3']['object']['key']
        print(key)
    role=get_execution_role()
    sess=sagemaker.Session()
    second_infer(bucket,key)



