import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def get_dataset():
    base_path = '/home/nadun/Documents/projects/flower/simulation_env/data/stack_overflow/'
    train = pd.read_csv(f'{base_path}/train.csv')[['label', 'text']]
    test = pd.read_csv(f'{base_path}/test.csv')[['label', 'text']]
    val = pd.read_csv(f'{base_path}/val.csv')[['label', 'text']]
    return train, test,  val

if __name__ == '__main__':
    train, test, val = get_dataset()

    print(train.head())