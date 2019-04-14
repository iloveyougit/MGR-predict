from flask import Flask, render_template, request

from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
sys.path.append(os.path.abspath('./model'))
from load import *

#init flask app
app = Flask(__name__)

global model, graph

model,graph= init()

@app.route('/')
def index():
    return render_template('index.html')

def convertImage(imgData):
    imgstr= re.search(r'base64,(.*'.imgdata1).group(1)
    with open('output.png','wb') as output:
        output.write(imgstr.decode('base64'))

@app.route('/predict',methods=['GET','POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('out.png', mode= 'L')
    x = np.invert(x)
    x = imresize(x,28,28)
    x = x.reshap(1,28,28,1)
    with graph.as_default():
        out = model.predict(x)
        respone = np.array_str(np.argmax(out))
        return respone

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
