import streamlit as st
import tensorflow as tf
import os
import pathlib
import cv2
import numpy as np
import zipfile

import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# 디렉토리 정보와 파일을 알려주면, 해당 디렉토리에
# 파일을 저장하는 함수를 만들겁니다
def save_uploaded_file(directory,file) :
    # 1. 디렉토리가 있는지 확인하여, 없으면 디렉토리부터 만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory,file.name),'wb') as f :
        f.write(file.getbuffer())
    return st.success('Saved file : {} in {}'.format(file.name,directory))

def main():
    st.title('Object Detection')
    uploaded_files=st.file_uploader('이미지 파일 업로드',type=['jpg','png','jpeg'],accept_multiple_files=True)
    
    if uploaded_files is not None :
        
        for file in uploaded_files :
            save_uploaded_file('temp_files',file)
            
    menu = ['ssd_mobilenet_v2_320x320_coco17_tpu-8','ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
            'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8','ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8']
    select_model = st.sidebar.radio('모델 선택',menu)
    if st.button('실행'):
        PATH_TO_LABELS = 'C:\\Users\\user12\\Documents\\TensorFlow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
    
        def download_model(model_name, model_date):
            base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
            model_file = model_name + '.tar.gz'
            model_dir = tf.keras.utils.get_file(fname=model_name,
                                                origin=base_url + model_date + '/' + model_file,
                                                untar=True)
            return str(model_dir)

        MODEL_DATE = '20200711'
        MODEL_NAME = select_model
        PATH_TO_MODEL_DIR = download_model(MODEL_NAME, MODEL_DATE)

        def load_model(model_dir):
            model_full_dir = model_dir + "/saved_model"
            detection_model = tf.saved_model.load(model_full_dir)
            return detection_model
            
        detection_model=load_model(PATH_TO_MODEL_DIR)

        
        
    # 우리가 가지고 있는 이미지 경로에서 이미지를 가져오는 코드
        PATH_TO_IMAGE_DIR = pathlib.Path('temp_files')
        IMAGE_PATHS = list( PATH_TO_IMAGE_DIR.glob('*.jpg') )
        
        # 이미지 경로에 있는 이미지를, 넘파이 행렬로 변경해주는 함수
        def load_image_into_numpy_array(path):   
            print(str(path))
            return cv2.imread(str(path))
        for image_path in IMAGE_PATHS:
    
            print('Running inference for {}... '.format(image_path), end='')



            image_np = load_image_into_numpy_array(image_path)
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

            st.image(image_np_with_detections,channels='BGR')
            
        
            
        
        
        
if __name__ == '__main__':
    main()