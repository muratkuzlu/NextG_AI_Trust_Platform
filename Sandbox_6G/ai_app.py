import streamlit as st
import pandas as pd
import numpy as np
import scipy.io
import altair as alt
from pathlib import Path
import os
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import run

loginSection = st.container()
headerSection = st.container()
logOutSection = st.container()
meinAppSection = st.container()

## Functions
@st.cache
def load_data(csvFile):
    df = pd.read_csv(csvFile)
    return df

@st.cache
def process_error_metrics(result):
    # Create table for RMSE, MSE, and MAE
    df = pd.DataFrame()
    df_graph = pd.DataFrame()
    #print(result['models'])
    lst_model_names = result['attacks']
    lst_fattack_rmse_list = result['attack_rmse_list']
    lst_eps_val =  result['epsilon']
    attack_title = result['epsilon_title']
   # st.write(lst_fattack_rmse_list)

    print(lst_model_names)
    print('#######################')
    print(lst_fattack_rmse_list)
    print('#######################')

    num_row = len(lst_eps_val)

    for i in range(0, num_row):
        # res_dct = {attack_title[0] : lst_eps_val[i]}
        res_dct = {lst_model_names[j]: lst_fattack_rmse_list[j*num_row + i] for j in range(0, len(lst_model_names))}
        #row = pd.Series(res_dct,name='RMSE')
        row = pd.Series(res_dct)
        df = df.append(row, ignore_index=True)

    df_graph = df
    # first_column) function
    df.insert(0, attack_title[0], lst_eps_val)
    df = df.set_index(attack_title[0])
    # Drop first column
    # df.drop(columns=df.columns[0], 
    #     axis=1, 
    #     inplace=True)
 


    return df, df_graph

#########################################################################################################################
@st.cache
def process_error_metrics_adv(result):
    # Create table for RMSE, MSE, and MAE
    df = pd.DataFrame()
    df = result['malicious_adv_outputs_list']

    return df

#########################################################################################################################
@st.cache
def process_prediction_results(result):
    # Plot actual and predictions
    df_predictions = pd.DataFrame()
    df_predictions['Actual'] = result['actual'].flatten().tolist()
    df_predictions['RNN_testPredict'] = result['RNN_testPredict'].flatten().tolist()
    df_predictions['LSTM_testPredict'] = result['LSTM_testPredict'].flatten().tolist()
    df_predictions['BiLSTM_testPredict'] = result['BiLSTM_testPredict'].flatten().tolist()
    df_predictions['GRU_testPredict'] = result['GRU_testPredict'].flatten().tolist()
    df_predictions['LSTM__Attention_testPredict'] = result['LSTM__Attention_testPredict'].flatten().tolist()
    df_predictions['BiLSTM__Attention_testPredict'] = result['BiLSTM__Attention_testPredict'].flatten().tolist()

    return df_predictions
#########################################################################################################################
@st.cache
def fetch_experiment_result(params):
    return run.run_experiment_1(params)
    #return 1

params = {}

#########################################################################################################################
def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

#########################################################################################################################
def show_main_page():
    with meinAppSection:

        st.title('NextG AI Trust Platform @ODU')


        col1, col2, col3 = st.columns(3)

        with col1:
            uploaded_file_model = st.sidebar.file_uploader("Load Model", type="hdf5")
            if uploaded_file_model is not None:
                # print(uploaded_file_model)
                param_model = uploaded_file_model.name
                # param_model = file_selector('Sandbox_6G')
                st.write("Model loaded")
                st.write(param_model)
                params['param_model'] = param_model
            else:
                params['param_model'] =''

            uploaded_input_data = st.sidebar.file_uploader("Input Data")
            if uploaded_input_data is not None:
                param_input_data = scipy.io.loadmat(uploaded_input_data)
                # st.write("Data loaded")
                params['param_input_data'] = param_input_data

            uploaded_output_data = st.sidebar.file_uploader("Output Data")
            if uploaded_output_data is not None:
                param_output_data = scipy.io.loadmat(uploaded_output_data)
                # st.write("Data loaded")
                params['param_output_data'] = param_output_data


            grid_search_selection = st.sidebar.radio('Grid search - Fine Tuning', ['False', 'True'])
            params['grid_search_selection'] = grid_search_selection
            # params['param_nr_epoch'] = st.sidebar.slider('Number of epochs', 0, 150, 1)
            # params['param_batch_size'] = st.sidebar.slider('Batch size', 8, 64, 32)

        with col2:
            # st.download_button('Download file', data)
            param_6G_application = st.sidebar.selectbox('6G Application', ['Beamforming', 'Channel Estimation', 'Intelligent Reflecting Surface (IRS)', 'Spectrum Sensing'])
            params['param_6G_application'] = param_6G_application

            param_attack_power = st.sidebar.selectbox('Attack Power', ['None', 'Low', 'Medium', 'High', 'All'])
            params['param_attack_power'] = param_attack_power


            param_attack_models = st.sidebar.multiselect('Attack Model',['Fast Gradient Sign Method (FGSM)', 
            'Basic Iterative Method (BIM)', 
            'Projected Gradient Descent (PGD)',
            'Momentum Iterative Method (MIM)',
            'Carlini & Wagner Attack (C&W)',
            'All'],
            default=['Fast Gradient Sign Method (FGSM)'])
            params['param_attack_models'] = param_attack_models


            param_defend_attack = st.sidebar.multiselect('Defend Attack', ['Adversarial Training' , 
                                                        'Defensive Distillation'],
                                                        default=['Adversarial Training'])
            params['param_defend_attack'] = param_defend_attack


        st.header('Experimental Results')

        if st.button('Run Experiment'):
            st.write('Running experiment ...')
            result = fetch_experiment_result(params)
            st.write('Done')

            # df_error_metrics, df_graph = process_error_metrics(result)
            df = process_error_metrics_adv(result)
            df_csv = df.to_csv().encode('utf-8')
            st.download_button(
            "Press to Download .csv format",
            df_csv,
            "file.csv",
            "text/csv",
            key='download-csv'
            )
            # # show table
            # st.write(df_error_metrics)
            # st.line_chart(df_error_metrics)

            # show table

            st.write(df)
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x='EPS', y='MSE',  hue='Attack')
            st.pyplot(fig)

# st.stop()
def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    
def show_logout_page():
    loginSection.empty()
    with logOutSection:
        st.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)
    
def LoggedIn_Clicked(userName, password):
    if userName == 'admin' and password=='admin':
        st.session_state['loggedIn'] = True
    else:
        st.session_state['loggedIn'] = False
        st.error("Invalid user name or password")

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            userName = st.text_input (label="", value="", placeholder="Enter your user name")
            password = st.text_input (label="", value="",placeholder="Enter password", type="password")
            st.button ("Login", on_click=LoggedIn_Clicked, args= (userName, password))

with headerSection:
    show_main_page() 
    # st.title("6G AI")
    #first run will have nothing in session_state
    # if 'loggedIn' not in st.session_state:
    #     st.session_state['loggedIn'] = False
    #     show_login_page() 
    # else:
    #     if st.session_state['loggedIn']:
    #         show_logout_page()    
    #         show_main_page()  
    #     else:
    #         show_login_page()
            
