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
#role = get_execution_role()
#print(role)
#sess = sagemaker.Session()
s3 = boto3.client('s3')

#origin_bucket = 'sro-ab3-data'
#key = '10-test-images/2resin.jpg'
output_bucket = 'sro-ab3-inference-output'
#unique_id = str(uuid.uuid1())

class ImageDeserializer(sagemaker.deserializers.BaseDeserializer):
    """Deserialize a PIL-compatible stream of Image bytes into a numpy pixel array"""
    def __init__(self, accept="image/png"):
        self.accept = accept

    @property
    def ACCEPT(self):
        return (self.accept,)

    def deserialize(self, stream, content_type):
        """Read a stream of bytes returned from an inference endpoint.
        Args:
            stream (botocore.response.StreamingBody): A stream of bytes.
            content_type (str): The MIME type of the data.
        Returns:
            mask: The numpy array of class labels per pixel
        """
        try:
            return np.array(Image.open(stream))
        finally:
            stream.close()

def visualize_detection(img_file, dets, classes=[], thresh=0.3):
            img = mpimg.imread(img_file)
            plt.imshow(img)
            height = img.shape[0]
            width = img.shape[1]
            colors = dict()
            print("{} detect was detected. See email with visuals".format(dets))
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

            print("Success... prepare to upload detected objects")
            output_prefix = 'defective'
            unique_id = str(uuid.uuid1())
            s3_predicted_location = '{}/{}-objects_detected.png'.format(output_prefix, unique_id)
            s3.upload_file('/tmp/predictedimage.png', output_bucket, s3_predicted_location)
            dets = None
            cls_id = None

def resize(newpath, width):
    im = Image.open(newpath)
    aspect = im.size[0] / im.size[1]
    resized = '/tmp/{}-test_resized.jpg'.format(unique_id)
    im.thumbnail([width, int(width / aspect)], Image.ANTIALIAS)
    im.save(resized, "JPEG")
    print('resize image done')
    return resized

def get_image(bucket,key):
    newpath = '/tmp/{}.jpg'.format(str(uuid.uuid1()))
    s3.download_file(bucket, key, newpath)
    print('get image done')
    return newpath

def second_infer(newpath):
    print("Starting second inference")
    f = None
    b = None
    results = None
    detections = None
    od_predictor = sagemaker.predictor.Predictor("ab3--od-inference")
    #resized = resize(newpath, 512)
    with open(newpath, 'rb') as image:
        f = image.read()
        b = bytearray(f)
        ne = open('/tmp/n.txt', 'wb')
        ne.write(b)
        results = od_predictor.predict(b, initial_args={'ContentType': 'image/jpeg'})
        detections = json.loads(results)
        object_categories = ['holes', 'resin', 'malformed']
        threshold = 0.2
        visualize_detection(newpath, detections['prediction'], object_categories, threshold)

def infer(bucket,key):
    ss_predictor = sagemaker.predictor.Predictor("ab3-ss-inference")
    newpath = get_image(bucket, key)
    #resized = resize(newpath, 640)
    ss_predictor.deserializer = ImageDeserializer(accept="image/png")
    ss_predictor.serializer = sagemaker.serializers.IdentitySerializer('image/jpeg')

    with open(newpath, 'rb') as imfile:
        imbytes = imfile.read()
        cls_mask = ss_predictor.predict(imbytes)
        print(type(cls_mask))
        print(cls_mask.shape)
        print(cls_mask)
    x = 0
    for l in cls_mask:
        for n in l:
            if 1 or 2 or 3 in n: 
            #for n in l:
                #if n == 1 or n == 2 or n == 3:
                #print("Defect Found")
                x = x + 1
                #run_defect(cls_mask, resized, newpath)
            else:
                pass
        print('X is {}'.format(x))
    if x == 0:
        run_acceptable()
    else:
        run_defect(cls_mask, newpath)
        #second_infer(newpath)
        #os.remove('/tmp/mask.png')
        #os.remove('/tmp/predictedimage.png')
        #os.remove('/tmp/pic.png')
        #os.remove('/tmp/transparent_mask.png')
        #os.remove('/tmp/highlight_mask.png')
        #os.remove('/tmp/newpath.jpg')
        #os.remove('/tmp/n.txt')


def run_defect(cls_mask, newpath):
    output_prefix = 'defective'
    plt.imshow(cls_mask, cmap='cool')

    plt.savefig('/tmp/mask.png')
    plt.show()
    im = Image.open(newpath)
    plt.imshow(im)
    plt.savefig('/tmp/pic.png')
    plt.show()

    im_rgb = Image.open('/tmp/mask.png')

    im_rgba = im_rgb.copy()
    im_rgba.putalpha(100)
    im_rgba.save('/tmp/transparent_mask.png')
    img = Image.open('/tmp/transparent_mask.png')
    background = Image.open('/tmp/pic.png')
    background.paste(img, (0, 0), img)
    background.save('/tmp/highlight_mask.png', "PNG")
    print("starting the upload mask upload...")
    unique_id = str(uuid.uuid1())
    s3_predicted_location = '{}/{}-pixel_map.png'.format(output_prefix, unique_id)
    s3.upload_file('/tmp/highlight_mask.png', output_bucket, s3_predicted_location)
    print("invoking 2nd inference")
    #second_infer(newpath)

def run_acceptable():
    output_prefix = 'acceptable'
    upload_resized = '/tmp/newpath.jpg'
    # background = Image.open(newpath)
    # background.save('/tmp/highlight_mask.png', "PNG")
    unique_id = str(uuid.uuid1())
    s3_predicted_location = '{}/{}-no_defects.png'.format(output_prefix, unique_id)
    s3.upload_file(upload_resized, output_bucket, s3_predicted_location)


def cleanup(bucket,key):
    s3 = boto3.resource('s3')
    copy_source = {
      'Bucket': bucket,
      'Key': key
    }
    cpbucket = s3.Bucket('sro-ab3-copied')
    cpbucket.copy(copy_source, key)
    s3 = boto3.client('s3')
    s3.delete_object(Bucket=bucket,Key=key)



def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        print(bucket)
        key = record['s3']['object']['key']
        print(key)
    role=get_execution_role()
    sess=sagemaker.Session()
    infer(bucket,key)
    cleanup(bucket,key)


