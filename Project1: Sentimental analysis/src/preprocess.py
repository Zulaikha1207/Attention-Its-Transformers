import pandas as pd
import numpy as np
import yaml
import os
from typing import Text
import argparse
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import seaborn as sns


def preprocess(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    print('Loading data..')
    df = pd.read_csv(config['data']['train_data_path'],  sep='\t')
    print(df.head())
    # Save raw data
    print('Training data loaded!')
    fig = plt.figure()
    sns.countplot(x="Sentiment", data=df)
    plt.savefig(config['preprocess']['preprocess_plot_path']) 
    plt.show()

    #define format for tokenization
    seq_len = config['preprocess']['seq_len']
    num_samples = len(df)
    print('Shape of tensor: ', num_samples, seq_len)
    
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # tokenize - this time returning Numpy tensors
    print('Tokenizing begin...')
    tokens = tokenizer(df['Phrase'].tolist(), max_length=seq_len, truncation=True,
                    padding='max_length', add_special_tokens=True,
                    return_tensors='np')
    print('Tokenizing completed...')
    print('Token tensors: ', tokens.keys())
    print('Token IDs', tokens['input_ids'][:10])

    ## Save token IDs and attention masks in nunpy vectors
    with open('movie-xids.npy', 'wb') as f:
        np.save(f, tokens['input_ids'])
    with open('movie-xmask.npy', 'wb') as f:
        np.save(f, tokens['attention_mask'])
    print('Token IDs and attention masks saved!')

    ## One hot encode the label column. Note that we're using tensors so input and output data has to be transformed into arrays/ tensors
    # first extract sentiment column
    arr = df['Sentiment'].values
    # we then initialize the zero array
    labels = np.zeros((num_samples, arr.max()+1))
    labels.shape
    labels[np.arange(num_samples), arr] = 1
    print('One hot encoded output labels: ', labels)
    with open('movie-labels.npy', 'wb') as f:
        np.save(f, labels)
    print('OHE Output labels saved!')


#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess(config_path=args.config)