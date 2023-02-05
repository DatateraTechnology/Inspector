from skimage import img_as_float
from quality import brisque as brisque
from quality import image_quality_matrix as imqm
from flask import Flask, jsonify, render_template, request,make_response
from azure.storage.blob import BlobServiceClient
import json
import requests
import io
import skimage
from os import path

app = Flask(__name__)

config = path.relpath("config/app-config.json")
with open(config, "r") as f:
    config = json.load(f)

def getQualityMatrixes():
    path = request.args.get('path')
    container = request.args.get('container')
    path = path + '/'+ container +'/'
    connect_str = config["blobStorage"]["imageQualityConnectionString"]
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_name = container
    container_client=blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()

    images = []
    for blob in blob_list:
        r = requests.get(path +  blob.name)
        stream = io.BytesIO(r.content)
        img = skimage.io.imread(stream, as_gray=True)
        if img is not None:
            img_float = img_as_float(img)
            score = brisque.score(img_float)
            obj = imqm.ImageQualityMatrix(blob.name, score)
            images.append(obj.__dict__)

    return images
