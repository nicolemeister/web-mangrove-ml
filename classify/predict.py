from datetime import datetime
import logging
import os

from urllib.request import urlopen
from PIL import Image
import tensorflow as tf
import numpy as np

import math
import pandas as pd
import gc


from tensorflow.keras.models import load_model


from .azure_blob import DirectoryClient


MAIN_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/"

IMAGE_DIRECTORY = MAIN_DIRECTORY + "images"

UPLOAD_FOLDER = IMAGE_DIRECTORY + '/images/'


CONTAINER_NAME = 'mvnmv4-merced' # Container name
CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=mangroveclassifier;AccountKey=s0T0RoyfFVb/Efc+e/s1odYn2YuqmspSxwRW/c5IrQcH5gi/FpHgVYpAinDudDQuXdMFgrha38b0niW6pHzIFw==;EndpointSuffix=core.windows.net'
MODEL_CONTAINER_NAME = 'mvnmv4-merced' # Container name

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


def download_model(client_model):
    client_model.download_file('mvnmv4-merced/saved_model.pb', MAIN_DIRECTORY + 'mvnmv4-merced/')
    client_model.download_file('mvnmv4-merced/variables/variables.data-00000-of-00002', MAIN_DIRECTORY + 'mvnmv4-merced/variables/')
    client_model.download_file('mvnmv4-merced/variables/variables.data-00001-of-00002', MAIN_DIRECTORY + 'mvnmv4-merced/variables/')
    client_model.download_file('mvnmv4-merced/variables/variables.index', MAIN_DIRECTORY + 'mvnmv4-merced/variables/')
    return 


scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

def _log_msg(msg):
    logging.info("{}: {}".format(datetime.now(),msg))

def _predict_image(image):
    try: 
        response = {
            'created': datetime.utcnow().isoformat(),
            'predictedTagName': 'hi',
            'prediction': 'hi'
        }

        _log_msg("Results: " + str(response))
        return response
            
    except Exception as e:
        _log_msg(str(e))
        return 'Error: Could not preprocess image for prediction. ' + str(e)


def get_batch_list(list_of_files, BATCH_SIZE):
    length = len(list_of_files)
    num_batches = math.floor(length/BATCH_SIZE) # number of batches with 32 files

    length_to_split = [BATCH_SIZE] * num_batches
    
    last_batch_length = length % BATCH_SIZE
    if last_batch_length != 0: 
        length_to_split.append(last_batch_length)

    from itertools import accumulate 
    batch_list = [list_of_files[x - y: x] for x, y in zip(accumulate(length_to_split), length_to_split)]
    return batch_list



def predict_image_from_url(start):

    BATCH_SIZE = 10
    BIG_BATCH_SIZE = 10 # how many total batches to do in this cycle

    start = int(start)

    url = 'https://mangroveclassifier.blob.core.windows.net/output-files/'

    output_container_name = 'output-files'
    client = DirectoryClient(CONNECTION_STRING, output_container_name)
    list_of_files = list(client.ls_files('', recursive=False))
    n_files = len(list_of_files)

    if n_files < start+(BATCH_SIZE*BIG_BATCH_SIZE):
        list_of_files = list_of_files[start:]
    else:
        list_of_files = list_of_files[start:start+(BATCH_SIZE*BIG_BATCH_SIZE)]

    # generate batches of 32 and download the files 32 at a time 
    batch_list = get_batch_list(list_of_files, BATCH_SIZE)
    # print(batch_list)

    #Set up dataframe that will hold classifications
    column_names = ["prediction","p_0","p_1","filename"]
    result_df = pd.DataFrame(columns=column_names)
    logging.info('created result_df')

    # load model
    model = MAIN_DIRECTORY + "mvnmv4-merced/"
    model = load_model(model)
    logging.info('loaded model:')

    prediction = []

    for n, batch in enumerate(batch_list):
        logging.info('starting batch: ')
        logging.info(n)
        # batch is a list of image names

        # create list of img urls 
        img_urls = []
        for img in batch:
            image_url = url + img
            img_urls.append(image_url)
        # logging.info('image urls: ')
        # logging.info(img_urls)

        # Create array of 32 images
        images = []
        for img_url in img_urls:
            with urlopen(img_url) as testImage:
                image = Image.open(testImage)
                image = np.asarray(image)
                images.append(image[:, :, :3])
        
        
        # scale images
        images = np.array(images)/255
        #logging.info('images:')
        #logging.info(images.shape)

        prediction.append(model.predict(images))

        '''prediction = model.predict(images)

        try:
            predictions
        except NameError:
            # prediction variable wasnt defined, create it
            predictions = prediction
        else:
            # prediction variable was defined, append to it
            predictions = np.concatenate((predictions, prediction), axis=0)
        '''
        gc.collect()


    # logging.info(predictions)
    predictions = np.concatenate(prediction, axis=0)

    #associate filenames and classification for each prediction
    for i,prediction in enumerate(predictions):
        # result_df.loc[i,"filename"] = data_gen.filenames[i]
        result_df.loc[i,"filename"] = list_of_files[i]

        #calculating predictions 
        result_df.loc[i,"p_0"] = sigmoid(prediction[0])
        result_df.loc[i,"p_1"] = sigmoid(prediction[1])
        
        #getting final class prediction
        result_df.loc[i,"prediction"] = np.argmax(prediction)

    # logging.info(result_df)

    gc.collect()

    response = result_df.to_json(orient="records")
    # logging.info(response)
    return response 

