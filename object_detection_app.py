import tensorflow as tf
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_util
import streamlit as st
from datetime import datetime

def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

def load_model(model_dir):
    model_full_dir = model_dir + "/saved_model"


    # Load saved model and build the detection function
    detection_model = tf.saved_model.load(model_full_dir)

    return detection_model
           
def run_object_detection_play():
    st.title('Tensorflow Object Detection')
    
    image_file = st.file_uploader("이미지를 업로드 하세요", type=['png','jpg','jpeg'])
    if image_file is not None :
        # 프린트문은 디버깅용으로서, 터미널에 출력한다.
        # print(type(image_file))
        # print(image_file.name)
        # print(image_file.size)
        # print(image_file.type)

        # 파일명을, 현재시간의 조합으로 해서 만들어보세요.
        # 현재시간.jpg
        thresh = st.sidebar.slider('모델 정확도 설정',0,100,value=50)
        st.subheader('모델 하나를 선택 해주세요')
        model_choice=st.radio('모델 선택',['ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8','ssd_mobilenet_v2_320x320_coco17_tpu-8','ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'])
        current_time = datetime.now()
        print(current_time)
        print(current_time.isoformat().replace(':', '_'))
        current_time = current_time.isoformat().replace(':', '_')
        image_file.name = current_time + '.jpg'

        img = Image.open(image_file)

        img_np = np.array(img)

   
        # 모델 다운로드 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        
        #http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

        LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        PATH_TO_LABELS = download_labels(LABEL_FILENAME)
        
        MODEL_DATE = '20200711'
        MODEL_NAME = model_choice
        PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
      

        detection_model = load_model(PATH_TO_MODEL_DIR)

        input_tensor = tf.convert_to_tensor(img_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detection_model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = img_np.copy()

        viz_util.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=thresh/100,
            agnostic_mode=False)
        
        st.image(image_np_with_detections)
        

        