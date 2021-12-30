import streamlit as st
import tensorflow as tf
import os
import pathlib

import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


def main():
    st.title('Object Detection')
    upload_file=st.file_uploader('이미지 파일 업로드',type=['jpg','png','jpeg'])
    
    if upload_file is not None :
        img=Image.open(upload_file)
        st.image(img)
    
        def download_model(model_name, model_date):
            base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
            model_file = model_name + '.tar.gz'
            model_dir = tf.keras.utils.get_file(fname=model_name,
                                                origin=base_url + model_date + '/' + model_file,
                                                untar=True)
            return str(model_dir)

        MODEL_DATE = '20200711'
        MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
        PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

        def load_model(model_dir):
            model_full_dir = model_dir + "/saved_model"
            detection_model = tf.saved_model.load(model_full_dir)
            return detection_model
            
        detection_model=load_model(PATH_TO_MODEL_DIR)

        def download_labels(filename):
            base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
            label_dir = tf.keras.utils.get_file(fname=filename,
                                                origin=base_url + filename,
                                                untar=False)
            label_dir = pathlib.Path(label_dir)
            return str(label_dir)

        LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        PATH_TO_LABELS = download_labels(LABEL_FILENAME)
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                            use_display_name=True)
        
        
        image_np=np.array(Image.open(upload_file))
        
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detection_model(input_tensor)
    #     print(detections)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        pil_image=Image.fromarray(image_np_with_detections)
        pil_image.show()
        st.image(pil_image)
            
        
        
        
        
        
        
        
        



if __name__ == '__main__':
    main()