import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

def get_dataset():
    file_name = '/home/nadun/Documents/projects/flower/datasets/movie-conversation.csv'
    data = pd.read_csv(file_name)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    test, val = train_test_split(test, test_size=0.2, random_state=42)

    return train, test,  val

if __name__ == '__main__':
    train, test, val = get_dataset()

    print(train.head())