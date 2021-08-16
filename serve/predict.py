import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

from utils import review_to_words, convert_and_pad

def model_fn(model_dir):
    '''
    Load the PyTorch model
    '''
    
    print("Loading model...")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Model loaded successfully")
    return model

def input_fn(serialized_input_data, content_type='text/plain'):
    '''
    Deserialize string input
    inputs:
        serialized_input_data: string, the data in string form
        content_type: string, which serializer to use. default='text/plain'
    '''
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    '''
    Serialize model output
    '''
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    '''
    Predict text sentiment
    inputs:
        input_data: string, text to predict the sentiment on
        model: LSTMClassifier, instanciated LSTMClassifier object  
    '''
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # transfer to device
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # clean the reviews text (text preprocessing)
    words = review_to_words(input_data)

    # convert the words to integers, replace infrequent words, and ensure the review is in uniform length
    data_X, data_len = convert_and_pad(model.word_dict, words)

    # Using data_X and data_len construct an appropriate input tensor
    data_pack = np.hstack((data_len, data_X)) # concatenate the length and the data
    data_pack = data_pack.reshape(1, -1)
    
    data = torch.from_numpy(data_pack)
    data = data.to(device)  # transfer to device

    # Make sure to put the model into evaluation mode
    model.eval()

    with torch.no_grad():
        pred = model(data)
   
    return float(pred.item())
