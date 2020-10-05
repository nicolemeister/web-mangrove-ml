import logging
import azure.functions as func
import json
import os
import gc

from .gdal_retile import retile

# Import helper script
#from .predict import classify
from .predict import predict_image_from_url

def main(req: func.HttpRequest) -> func.HttpResponse:
    gc.collect()

    method = str(req.params.get('method'))
    logging.info('Method received: ' + method)

    start = str(req.params.get('start'))
    logging.info('start received: ' + start)

    '''name = str(req.params.get('name'))
    logging.info('name: ' + name)'''

    logging.info('os cwd: ' + os.getcwd())


    if method == 'predict':
        results = predict_image_from_url(start)
    if method == 'retile':
        # results = retile(name)
        results = retile(TargetDir_=os.getcwd() + "/classify/images",  Names_= [os.getcwd() + "/classify/lap_2018-07_site05_120m_RGB_cc.tif"])
    # results = classify()

    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    return func.HttpResponse(json.dumps(results), headers = headers)
