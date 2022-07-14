import numpy as np
import streamlit as st 
from PIL import Image
import cv2
from os import listdir
from os.path import join, isfile
import requests

import test
from test import main_test
url = 'http://192.168.20.156:5000/detect2ocr'

st.title("About")
st.info("This is a demo application written to help you extract automatically extract information from Vietnamese receipt")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
col1, col2 = st.columns(2)
with col1:
    if uploaded_file is not None:
        temp = Image.open(uploaded_file).convert('RGB') 
        image=np.array(temp)
        image = image[:, :, ::-1].copy() 
        cv2.imwrite('/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/filename.jpeg',image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        with col2:
                r = requests.post(url)
                main_test()
                st.balloons()
                with open('/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/output/filename.txt', 'r') as fi:
                    lines = fi.read().splitlines()
                results={}
                for line in lines:
                    temp=line.partition("\t")
                    if temp[0] in results:
                        continue
                    else:
                        if temp[0]=='TOTAL_COST':
                            if temp[2][1].isnumeric():
                                results[temp[0]]=temp[2]
                            else:
                                continue
                        results[temp[0]]=temp[2]
                for key in results:
                    st.info(f'{key}: {results[key]}')

