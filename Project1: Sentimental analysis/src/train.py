import pandas as pd
import numpy as np
import yaml
import os
from typing import Text
import argparse
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import seaborn as sns
import tensorflow as tf
from transformers import TFAutoModel

def train(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    with open('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/movie-xids.npy', 'rb') as f:
        Xids = np.load(f, allow_pickle=True)
    with open('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/movie-xmask.npy', 'rb') as f:
        Xmask = np.load(f, allow_pickle=True)
    with open('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/movie-labels.npy', 'rb') as f:
        labels = np.load(f, allow_pickle=True)

    #Convert three arrays and into TF dataset object using from_tensor_slices
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    print('Tensorflow dataset object showing 1 observation: ', dataset.take(1))

    # Define map function that splits the input tensors from ouput 
    def map_func(input_ids, masks, labels):
        # we convert our three-item tuple into a two-item tuple where the input item is a dictionary
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    # then we use the dataset map method to apply this transformation
    dataset = dataset.map(map_func)
    print('TF object after splitting input and output tensors: ', dataset.take(1))

    #create batches and shuffle data
    batch_size = config['train']['batch_size']
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    dataset.take(1)

    #split data into training and validation sets using take and skip methods. Creating a 90-10 split
    split = config['train']['split']
    size = int((Xids.shape[0] / batch_size) * split)
    print('Performing 90 by 10 split! The training data shape is: ', size)

    #create train and validation set
    train_ds = dataset.take(size)
    val_ds = dataset.skip(size)

    #free up space
    del dataset

    #save train and validation sets
    tf.data.experimental.save(train_ds, '/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/train')
    tf.data.experimental.save(val_ds, '/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/val')

    #toload the train and val set we need the element spec of the sets. This signifies the shape and format of the tensors
    print('The element spec of the train and validation set: ', train_ds.element_spec)

    ##Building the model
    # initialize the Bert model, loaded as a pretrained model from transformers
    bert = TFAutoModel.from_pretrained('bert-base-cased')
    print('Summary of the native pre-trained BERT model (without any adjustments made to the architecture): ', bert.summary())

    """Define architecture around the BERT, as follows:
    - Two input layers (one for input IDs and one for attention mask).
    - A post-bert dropout layer to reduce the likelihood of overfitting and improve generalization.
    - Max pooling layer to convert the 3D tensors output by Bert to 2D.
    - Final output activations using softmax for outputting categorical probabilities."""

    #define model architecture
    # two input layers, we ensure layer name variables match to dictionary keys in TF dataset
    input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')

    #embedding layer
    embeddings = bert.bert(input_ids, attention_mask = mask)[1]  #access final activations (alread max-pooled) [1]

    #convert bert embeddings into 5 output classes
    x =tf.keras.layers.Dense(1024, activation='relu')(embeddings)
    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)

    #define our model
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    #freeze the Bert layer because Bert is already highly trained, and contains huge number of parameters (computation and time needed exhausting)
    model.layers[2].trainable = False
    print('Architecture of the bert model', model.summary())

    #initialise training paramters and optimizers
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    ##train model using train and validation tensors
    # load the training and validation sets
    train_ds = tf.data.experimental.load('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/train', element_spec=train_ds.element_spec)
    val_ds = tf.data.experimental.load('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/val', element_spec=train_ds.element_spec)

    # view the input format
    print('Input format of the model: ', train_ds.take(1))

    history = model.fit(train_ds, validation_data=val_ds,epochs=1)
    model.save('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/results/sentiment_model')

#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)