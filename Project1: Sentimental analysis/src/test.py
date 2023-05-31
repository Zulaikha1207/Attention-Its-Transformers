import pandas as pd
import numpy as np
import yaml
import os
from typing import Text
import argparse
import tensorflow as tf
from transformers import BertTokenizer

def test(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    model = tf.keras.models.load_model('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/results/sentiment_model')
    print('Summary of model: ', model.summary())

    """Before making predictions we need to format our data, which requires two steps:
    - Tokenizing the data using the bert-base-cased tokenizer.
    - Transforming the data into a dictionary containing 'input_ids' and 'attention_mask' tensors."""

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def prep_data(text):
        tokens = tokenizer.encode_plus(text, max_length=512,
        truncation=True, padding='max_length', add_special_tokens=True,
        return_token_type_ids= False, return_tensors='tf')

        return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
            'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

    #testing with hello world text
    probs = model.predict(prep_data('hello world'))[0]
    print('Model output class when given hello world as input text: ', np.argmax(probs))

    # making predictions on test set
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv('/Users/zulikahlatief/Desktop/personal/NLP/Project1: Sentimental analysis/data/test.tsv', sep='\t')
    print('Test data set: ', df.head())
    df = df.drop_duplicates(subset=['SentenceId'], keep='first')

    #make predictions
    df['Sentiment'] = None

    for i, row in df.iterrows():
        # get token tensors
        tokens = prep_data(row['Phrase'])
        # get probabilities
        probs = model.predict(tokens)
        # find argmax for winning class
        pred = np.argmax(probs)
        # add to dataframe
        df.at[i, 'Sentiment'] = pred

    print('Sentiments predicted on test set: ', df.head())
    print('Sentiments predicted on test set: ', df.tail())

#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    test(config_path=args.config)