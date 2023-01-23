import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import time
from plot import display_image, download_and_resize_image, draw_bounding_box_on_image, draw_boxes


def download_model(name):
    if name == 'inception':
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    elif name == 'mobilenet':
        module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    model = hub.load(module_handle)
    detector = model.signatures['default']
    return detector
    
    
def load_img(path):
    '''
    Loads a JPEG image and converts it to a tensor.

    Args:
        path (string) -- path to a locally saved JPEG image

    Returns:
        (tensor) -- an image tensor
    '''

    # read the file
    img = tf.io.read_file(path)

    # convert to a tensor
    img = tf.image.decode_jpeg(img, channels=3)

    return img


def run_detector(detector, img, max_boxes=10, min_score=0.1, plot_info=True):
    '''
    Runs inference on a local file using an object detection model.

    Args:
        detector (model) -- an object detection model loaded from TF Hub
        img (np.array) -- image
    '''

    # load an image tensor from a local file path
    img = np.asarray(img)
    
    img = tf.convert_to_tensor(img)

    # add a batch dimension in front of the tensor
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    # run inference using the model
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    # save the results in a dictionary
    result = {key: value.numpy() for key, value in result.items()}

    # print results
    if plot_info:
        print("Found %d objects." % len(result["detection_scores"]))
        print("Inference time: ", end_time - start_time)

    # draw predicted boxes over the image
    image_with_boxes = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"], min_score=min_score, max_boxes=max_boxes,
        plot_info=plot_info)

    # display the image
    return image_with_boxes