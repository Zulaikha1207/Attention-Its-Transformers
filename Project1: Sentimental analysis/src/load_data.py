import pandas as pd
import numpy as np
import yaml
import os
from typing import Text
import argparse

def load_data(config_path: Text) -> None:
    with open("params.yaml") as config_file:
        config = yaml.safe_load(config_file)
    
    print('Loading data..')
    dataset = pd.read_csv(config['data']['train_data_path'],  sep='\t')
    print(dataset.head())
    # Save raw data
    print('Training data load complete!')

#to run from CLI use a constructer that allows to parse config file as an argument to the data_load function
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    load_data(config_path=args.config)