import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import date,datetime
import numpy as np

from object_detection_app import run_object_detection_play

# 디렉토리 정보와 파일을 알려주면, 해당 디렉토리에
# 파일을 저장하는 함수를 만들겁니다.
def save_uploaded_file(directory, file) :
    # 1.디렉토리가 있는지 확인하여, 없으면 디렉토리부터만든다.
    if not os.path.exists(directory) :
        os.makedirs(directory)
    # 2. 디렉토리가 있으니, 파일을 저장.
    with open(os.path.join(directory, file.name), 'wb') as f :
        f.write(file.getbuffer())
    return st.success("Saved file : {} in {}".format(file.name, directory))


def main():

    menu = ['Object Detection', 'About']

    choice = st.sidebar.selectbox('메뉴 선택', menu)

    if choice == 'Object Detection' :
        run_object_detection_play()


    

if __name__ == '__main__' :
    main()