#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:17:00 2022

@author: ozgur
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tqdm.notebook import tqdm
from loguru import logger
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tqdm.keras import TqdmCallback
from plot_keras_history import plot_history
import seaborn as sns

np.random.seed(10)



def run_all(params):
    
    print("Run ALL")

    result = {}
    #########################################################################
    ### CONFIGURATION
    array_6G_application = params['param_6G_application']
    array_attack_models = params['param_attack_models']
    attack_power = params['param_attack_power']
    array_defend_attack = params['param_defend_attack']
    loaded_model = params['param_model']

    # run_epochs = params['param_nr_epoch'] #100
    # run_batch_size = params['param_batch_size'] #32

#########################################################################
################## ATTACK MODELS  #######################################
    list_attack_models = ['Fast Gradient Sign Method (FGSM)', 
                        'Basic Iterative Method (BIM)', 
                        'Projected Gradient Descent (PGD)',
                        'Momentum Iterative Method (MIM)',
                        'Carlini & Wagner Attack (C&W)']
    
    run_attack_models = []
    if list_attack_models[0] in array_attack_models:
        run_attack_models.append("FGSM")
    if list_attack_models[1] in array_attack_models:
        run_attack_models.append("BIM")
    if list_attack_models[2] in array_attack_models:
        run_attack_models.append("PGD")
    if list_attack_models[3] in array_attack_models:
        run_attack_models.append("MIM")
    if list_attack_models[4] in array_attack_models:
        run_attack_models.append("CW")
    
    if run_attack_models == []:
        print("There is no attack model for running")
    else:
        print("Run Attack Models:")
        print(run_attack_models)


#########################################################################
################## ATTACK POWER  #######################################
    list_attack_power = ['None', 'Low', 'Medium', 'High', 'All']
    run_eps_val = []
    if list_attack_power[0] in attack_power:
        run_eps_val.append(0)
    if list_attack_power[1] in attack_power:
        run_eps_val = [0.25]
    if list_attack_power[2] in attack_power:
        run_eps_val = [0.75]
    if list_attack_power[3] in attack_power:
        run_eps_val = [1.0]
    if list_attack_power[4] in attack_power:
        run_eps_val = [0, 0.25, 0.5, 0.75, 1.0]
        # for i in range(1,11,4):
        #     run_eps_val.append(float(i)/10)

    print("Attack Power")
    print(run_eps_val)


#########################################################################
################## DEFENF MODELS  #######################################
    list_defend_models = ['Adversarial Training', 
                        'Defensive Distillation']
    
    run_defend_models = []
    if list_defend_models[0] in array_defend_attack:
        run_defend_models.append("Adversarial Training")
    if list_attack_models[1] in array_defend_attack:
        run_defend_models.append("Defensive Distillation")
    
    if run_defend_models == []:
        print("There is no defend model for running")
    else:
        print("Run Defend Models:")
        print(run_defend_models)

    ### CONFIGURATION END
    #########################################################################

    In_train, In_test, Out_train, Out_test = get_dataset()
    model = keras.models.load_model('Sandbox_6G/ex/O1_bs_0.hdf5')
    #model = keras.models.load_model(loaded_model)
    #model = loaded_model
    
    ## generate malicious inputs
    attack_power_list = []
    attack_name_list = []
    malicious_outputs_list = []
    adv_malicious_outputs_list = []
    malicious_adv_outputs_list = []
    for attack_name in run_attack_models:
        for i in range(len(run_eps_val)):
            model = keras.models.load_model('Sandbox_6G/ex/O1_bs_0.hdf5')
            #In_test_adv = attack_models(model,attack_name='PGD',eps_val=0.1,testset=In_test)
            In_test_adv = attack_models(model,attack_name=attack_name,eps_val=run_eps_val[i],testset=In_test)
            malicious_outputs = model.predict(In_test_adv)
            mse_malicious = mean_squared_error(Out_test[:,0:512],malicious_outputs)
            #malicious_outputs_list.append(run_eps_val[i])
            attack_power_list.append(run_eps_val[i])
            attack_name_list.append(attack_name+'_undefended')
            malicious_outputs_list.append(mse_malicious)
            #model, df_adv_train = adv_train(model,In_test,Out_test[:,0:512], epsilon=run_eps_val[i], attack_lists=attack_name,adv_train_steps=2, plot_train_history=False)
            if run_defend_models != []:
                model, df_adv_train = adv_train(model,In_test,Out_test[:,0:512], run_eps_val[i], attack_name ,adv_train_steps=10, plot_train_history=False)
                adv_MSE = df_adv_train.MSE.values[9]
                attack_power_list.append(run_eps_val[i])
                attack_name_list.append(attack_name+'_defended')
                malicious_outputs_list.append(adv_MSE)

            
    df = pd.DataFrame({'Attack':attack_name_list,'EPS':attack_power_list,
                    'MSE':malicious_outputs_list})

    print("############################################################")
    print('Apply ' + str(attack_name) + ".....")
    print('Attack Power ' + str(run_eps_val) + ".....")
    print(malicious_outputs_list)
    print("############################################################")
    result['epsilon']= run_eps_val
    result['epsilon_title']= ['EPS']
    result['attacks']= run_attack_models
    result['attack_rmse_list'] = malicious_outputs_list
    result['adv_malicious_outputs_list'] = adv_malicious_outputs_list
    result['malicious_adv_outputs_list'] = df
    return result


def attack_models(model, attack_name, eps_val, testset, norm=np.inf):
    logger.debug("Attack started " + attack_name)
    logits_model = tf.keras.Model(model.input, model.layers[-1].output)
    if attack_name == 'FGSM':
        In_test_adv = fast_gradient_method(logits_model, testset, eps_val,
                                           norm, targeted=False,
                                           clip_min=testset.min(),clip_max=testset.max())
        return In_test_adv
    elif attack_name == 'PGD':
        In_test_adv = projected_gradient_descent(logits_model, testset, eps=eps_val, 
                                         norm=norm,nb_iter=50,eps_iter=eps_val/10.0,
                                         targeted=False,
                                         clip_min=testset.min(),clip_max=testset.max())
        return In_test_adv
    elif attack_name == 'BIM':
        In_test_adv = basic_iterative_method(logits_model,testset,eps_val,
                                             eps_iter=eps_val/10.0, nb_iter=50,norm=norm,
                                             targeted=False)
        return In_test_adv
    # elif attack_name == 'MIM':
    #     In_test_adv = momentum_iterative_method(logits_model,testset,eps_val,
    #                                          eps_iter=eps_val/10.0, nb_iter=50,norm=norm,
    #                                          targeted=False,
    #                                          clip_min=testset.min(),clip_max=testset.max())
    #     return In_test_adv
    elif attack_name == 'CW':
        In_test_adv = []
        for i in tqdm(range(100)):
            tmp = carlini_wagner_l2(logits_model, testset[i:i+1,:].astype(np.float32),
                                    targeted=False,# y=[np.float64(0.0)],
                                    batch_size=512, confidence=100.0,
                                    abort_early=False, max_iterations=100,
                                    clip_min=testset.min(),clip_max=testset.max())
            In_test_adv.append(tmp)
        
        return np.array(In_test_adv)
    elif attack_name == 'MIM':
        print('*'*20)
        In_test_adv = momentum_iterative_method(logits_model,testset,
                                                eps=eps_val,eps_iter=0.2,
                                                nb_iter=50,norm=norm)
        return In_test_adv
    
def adv_train(model, X_test, Y_test, epsilon, attack, adv_train_steps=10, plot_train_history=False):
    mse_vals_list = []
    attack_name_list = []
    iter_list = []
    
    for i in tqdm(range(adv_train_steps), position=0, leave=True):
        # for attack in attack_lists:
        adv_inputs = attack_models(model, attack, epsilon, X_test)
        adv_inputs_pred = model.predict(adv_inputs)
        mse_vals = []
        for j in range(adv_inputs_pred.shape[0]):
            tmp = mean_squared_error(Y_test[j,:], adv_inputs_pred[j,:])
            mse_vals.append(tmp)
            
        In_train_adv = np.concatenate((X_test, adv_inputs), axis=0)
        Out_train_adv = np.concatenate((Y_test, Y_test), axis=0)
        
        es = EarlyStopping(monitor='val_loss', patience=10, verbose=1,
                        restore_best_weights=True, mode='min')
        
        hist = model.fit(In_train_adv,
                    Out_train_adv,
                    batch_size=300,
                    epochs=500,
                    verbose=0,
                    validation_data=(adv_inputs, Y_test),
                    callbacks=[es,TqdmCallback(verbose=1)])
            
            # if plot_train_history:
            #     plot_history(hist.history)
            #     plt.show()
            
            
                
        mse_vals_list.append(np.mean(mse_vals))
        attack_name_list.append(attack)
        iter_list.append(i)
            
    df = pd.DataFrame({'Attack':attack_name_list,'Iter_No':iter_list,
                       'MSE':mse_vals_list})
    
    return model, df

def get_dataset():
    In_set_file=loadmat('Sandbox_6G/ex/O1_DLCB_input.mat')
    Out_set_file=loadmat('Sandbox_6G/ex/O1_DLCB_output.mat')
    
    In_set=In_set_file['DL_input']
    Out_set=Out_set_file['DL_output']
    
    In_set_real = np.zeros((In_set.shape[0], In_set.shape[1]*2))
    for i in range(In_set.shape[0]):
        for j in range(In_set.shape[1]):
            In_set_real[i,j*2] = np.real(In_set[i,j])
            In_set_real[i,j*2+1] = np.imag(In_set[i,j])
            
    num_user_tot=In_set.shape[0]
    
    In_train, In_test, Out_train, Out_test = train_test_split(In_set_real, Out_set, test_size=0.2)
    return In_train, In_test, Out_train, Out_test

