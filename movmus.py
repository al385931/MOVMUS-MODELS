
import pandas as pd
import numpy as np
import os
import h5py
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import json

''' GENERAL FUNCTIONS '''

'''DATA FILE FUNCTIONS'''
# This function is to save test and training data into a .npz file
def save_data_npz(training_data, validation_data, test_data, filename, path='.\\DATA'):
    # Check if the path exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the data to the specified path
    np.savez(os.path.join(path, filename), training_data=training_data, validation_data=validation_data, test_data=test_data)
    print(f"Data saved to {os.path.join(path, filename)}")

# This function is to load test and training data from a .npz file
def load_data_npz(filename, path='.\\DATA'):
    # Load the data from the specified path
    data = np.load(os.path.join(path, filename))
    print(f'shape of training data: {data["training_data"].shape}')
    print(f'shape of validation data: {data["validation_data"].shape}')
    print(f'shape of test data: {data["test_data"].shape}')

    return data['training_data'], data['validation_data'], data['test_data']

def transpose_check_data(training_data, validation_data, test_data):
    transposed_training_data = training_data.transpose(0, 2, 1)
    transposed_validation_data = validation_data.transpose(0, 2, 1)
    transposed_test_data = test_data.transpose(0, 2, 1)

    print(f"Training data shape: {transposed_training_data.shape}")
    print(f"Validation data shape: {transposed_validation_data.shape}")
    print(f"Test data shape: {transposed_test_data.shape}")
    if transposed_training_data.shape[1:] == transposed_test_data.shape[1:] == transposed_validation_data.shape[1:]:
        print("Shapes match")
    else:
        print("Shapes do not match")

    return transposed_training_data, transposed_validation_data, transposed_test_data

'''LABELS FILE FUNCTIONS'''
# this function filters the labelled df to match the set
def filter_by_ids(id_list, target_df, id_column='ID'):
    return target_df[target_df[id_column].isin(id_list)]


# This function is to prepare the labels for the model

def load_labels(filename, path='.\\DATA'):
    # Load the labels from the specified path
    with np.load(os.path.join(path, filename), allow_pickle=True) as data:
        
        loaded_data = data['data']
        columns = data['columns']
        index = data['index']

    df = pd.DataFrame(data=loaded_data, columns=columns, index=index)
    df.head()
    return df

def adjust_labels(df):
    labels = (df['GRASP_DH'].apply(lambda x: 0 if x != 1 else 1)).astype(int)  # Shape: (3440,)
    num_classes =2  # There are 8 different types of grasp
    labels_categorical = to_categorical(labels, num_classes)
    print("Labels after one-hot encoding:", labels_categorical.shape)
    return labels_categorical


''' METADATA FILE FUNCTIONS '''


# This function is to save a dataframe to a .npz file
# p.e: save_df_to_npz(target_df, 'target_df.npz', path='.\\DATA')

def save_metadata_labels_npz(df, filename, path='.\\DATA'):
    # Check if the path exists
    if not os.path.exists(path):
        os.makedirs(path)   
    # Save the dataframe to the specified path
    np.savez(os.path.join(path, filename), 
             data=df.values, 
             columns=df.columns, 
             index=df.index)
    print(f"DataFrame saved to {os.path.join(path, filename)}")

# This function is to load a dataframe from a .npz file
# p.e: df = load_metadata_npz('target_df.npz', path='.\\DATA')

def load_metadata_npz(filename, path='.\\DATA'):
    # Load the dataframe from the specified path
    data = np.load(os.path.join(path, filename), allow_pickle=True)
    return pd.DataFrame(data=data['data'], columns=data['columns'], index=data['index'])

# this function returns a dataframe with the rows that match the specified values
# IMPORTANT you must pass the values as a list
# for example: selected_df = select_rows(df, participant_list=[1, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26], task_height_list=[4,2])


def select_rows(df, participant_list=None, task_height_list=None, id_list=None, task_list=None, et_list=None, grasp_dh_list=None):
    if participant_list is not None:
        df = df[df['PARTICIPANT'].isin(participant_list)]

    if task_height_list is not None:
        df = df[df['TASK_HEIGHT'].isin(task_height_list)]

    if task_list is not None:
        df = df[df['T'].isin(task_list)]
    
    if et_list is not None:
        df = df[df['ET'].isin(et_list)]

    if grasp_dh_list is not None:
        df = df[df['GRASP_DH'].isin(grasp_dh_list)]
    
    if id_list is not None:
        df = df[df['ID'].isin(id_list)]
    return df


# this function returns a list of unique IDs
def get_ids(df, id_column='ID'):
    return df[id_column].unique().tolist()



''' MODELS FUNCTIONS '''
def save_model(model, model_folder, model_name):
    model.save(f'{model_folder}\\{model_name}.keras')
    print(f"Model saved to {model_folder}\\{model_name}.keras")

def load_model(model_folder, model_name):
    print(f"Loading model from {model_folder}\\{model_name}.keras")
    
    m = tf.keras.models.load_model(f'{model_folder}\\{model_name}.keras')
    check_model_layers(m)
    return m

def save_history(history, model_folder, model_name):
    with open(f'{model_folder}\\{model_name}_history.json', 'w') as f:
        json.dump(history, f)
    print(f"History saved to {model_folder}\\{model_name}_history.json")

def load_history(model_folder, model_name):
    with open(f'{model_folder}\\{model_name}_history.json', 'r') as f:
        print(f"History loaded from {model_folder}\\{model_name}_history.json")
        history = json.load(f)
    return history

def check_model_layers(model):
    # Iterate through layers and print configurations
    print("Num Layers:", len(model.layers))
    print("----")
    #print("Model layers:", model.layers)
    for layer in model.layers:
        print("----")
        print("Layer:", layer.name)
        config = layer.get_config()
        print(json.dumps(config, indent=4))
        print("----")
        
    '''for layer in model.layers:
        # Check if the layer is a Conv1D layer
        if isinstance(layer, tf.keras.layers.Conv1D):
            print(f"Layer: {layer.name}")
            print(f"Kernel Size: {layer.kernel_size}")
            print(f"Filters: {layer.filters}")
            print(f"Activation: {layer.activation.__name__}")
            print(f"Padding: {layer.padding}")
            print("----")
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            print(f"Layer: {layer.name}")
            print("----")
        
        # Check if the layer is a MaxPooling1D layer
        if isinstance(layer, tf.keras.layers.MaxPooling1D):
            print(f"Layer: {layer.name}")
            print(f"Pool Size: {layer.pool_size}")
            print("----")
        
        # gap layer
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            print(f"Layer: {layer.name}")
            print("----")

        # dropout layer
        if isinstance(layer, tf.keras.layers.Dropout):
            print(f"Layer: {layer.name}")
            print(f"Rate: {layer.rate}")
            print("----")

        # For Dense layers
        if isinstance(layer, tf.keras.layers.Dense):
            print(f"Layer: {layer.name}")
            print(f"Units: {layer.units}")
            print(f"Activation: {layer.activation.__name__}")
            print("----")
        '''