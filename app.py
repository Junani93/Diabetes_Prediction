import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
from eda_app import run_eda_app
from ml_app import run_ml_app
from sklearn.ensemble import RandomForestClassifier

def main() :
    model = joblib.load('data/randomforest.pickle')
    df = pd.read_csv('data/diabetes.csv')
    st.dataframe(df)

    new_data = np.array([3, 88, 58, 11, 54, 24])
    new_data = new_data.reshape(1,-1)
    print(new_data)

    st.write(model.predict(new_data)
    # st.title('Predict Diabetes')

    # # 사이드바 메뉴
    # menu = ['Home', 'EDA', 'ML']
    # choice = st.sidebar.selectbox('Menu', menu)

    # if choice == 'Home' :
    #     st.write('이 앱은 데이터를 가지고 당뇨병과의 상관관계를 분석한 내용입니다. 정보를 입력하면, 당신의 당뇨병의 정도를 에측하는 APP입니다.')
    #     st.write('왼쪽의 사이드바에서 선택하세요.')
    # elif choice == 'EDA' :
    #     run_eda_app()
    # elif choice == 'ML' :
    #     run_ml_app()

if __name__ == '__main__' :
    user_name_list = []
    main()