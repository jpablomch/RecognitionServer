#!/usr/bin/env python

import csv
import cv2
import caffe
from caffe.proto import caffe_pb2
import locale
import numpy as np
import pandas as pd
from dropbox import client, rest, session
import os
import sys

def img_equ(image):
    for i in [0, 1, 2]:
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    image = cv2.resize(image, (227, 227), interpolation = cv2.INTER_CUBIC)
    return image

# Using OAUTH2
oauth2_token = 'INSERT YOUR TOKEN HERE'
MODEL_DIRNAME = 'INSERT PATH HERE'
meanDataFile = '{}/NAME OF MEAN FILE'.format(MODEL_DIRNAME)
modelDeploymentFile = '{}/DEPLOY FILE'.format(MODEL_DIRNAME)
model = '{}/MODEL NAME'.format(MODEL_DIRNAME)
classesFile = '{}/ CLASSES '.format(MODEL_DIRNAME)
pathWD = ''

dropboxClient = client.DropboxClient(oauth2_token) #TODO: Use new API

caffe.set_mode_gpu()
meanBlob = caffe_pb2.BlobProto()
mean = None
with open(meanDataFile) as f:
    meanBlob.ParseFromString(f.read())
    mean = np.asarray(meanBlob.data, dtype=np.float32).reshape(
        (meanBlob.channels, meanBlob.height, meanBlob.width))
netJP = caffe.Classifier(
    modelDeploymentFile, model,
    caffe.TEST
)
dataTransformer = caffe.io.dataTransformer({'data' : netJP.blobs['data'].data.shape})
dataTransformer.set_mean('data', mean)
dataTransformer.set_transpose('data', (2, 0, 1))
with open(classesFile) as f:
    labelsDataframe = pd.DataFrame([
        {
            'id':l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
    ])
labels = labelsDataframe.sort('id')['name'].values
# Check pictures and text files
dropboxResponse = dropboxClient.metadata('/UserPicturesToProcess')
listOfPending = []
encoding = locale.getdefaultlocale()[1] or 'ascii'
if 'contents' in dropboxResponse:
    for f in dropboxResponse['contents']:
        name = os.path.basename(f['path'])
        listOfPending.append(name)
    print(listOfPending)

already_downloaded = [f for f in os.listdir('LocalProcessedImages/')]
print (already_downloaded)
for item in listOfPending:
    if item not in already_downloaded:
        print ('Downloading ' + item)
        with open(('LocalProcessedImages/' + item).encode(encoding), "wb") as to_file:
            f, metadata = dropboxClient.get_file_and_metadata('/UserPicturesToProcess/' + item)
            to_file.write(f.read())
print("Recognizing...")
already_results = [f[:-4] for f in os.listdir('LocalResults/')]
print(already_results)
for item in listOfPending:
    if item[:-4] not in already_results:
        image = cv2.imread('LocalProcessedImages/' + item, cv2.IMREAD_COLOR)
        image = img_equ(image)
        netJP.blobs['data'].data[...] = dataTransformer.preprocess('data', image)
        out = netJP.forward()
        probs = out['prob']
        scores = probs.flatten()
        indices = (-scores).argsort()[:5]
        predictions = labels[indices]

        meta = [
            (p, '%.5f' % scores[i])
            for i, p in zip(indices, predictions)
        ]

        with open("LocalResults/" + item[:-3] + "csv", 'wb') as resultFile:
            print("Recognizing " + item)
            wr = csv.writer(resultFile, dialect='excel')
            for i in xrange(len(meta)):
                wr.writerow([meta[i][0], meta[i][1]])

        # Copy local file to Dropbox results folder
        fromFile = open("LocalResults/" + item[:-3] + "csv", "rb")
        fullPath = ("/RecognitionResults/" + item[:-3] + "csv").decode(encoding)
        dropboxClient.put_file(fullPath, fromFile)
        # Validate and move to different folder

        # Create thumbnail.
        thumbnailWidth = 200 if image.shape[0] > image.shape[1] else 200*(image.shape[0]/image.shape[1])
        thumbnailHeight = 200 if image.shape[1] > image.shape[0] else 200*(image.shape[1]/image.shape[0])

        # print(image.shape[0],  image.shape[1])
        # print(thumbnailWidth, thumbnailHeight)
        thumbnail = cv2.resize(image, (thumbnailWidth, thumbnailHeight), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("LocalThumbnails/" + item, thumbnail)
        fromFile = open("LocalThumbnails/" + item, "rb")
        fullPath = ("/RecognitionProcessedThumbnails/" + item).decode(encoding)
        dropboxClient.put_file(fullPath, fromFile)