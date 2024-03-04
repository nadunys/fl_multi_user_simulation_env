import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class Data():
    def __init__(self, test_split: float, val_split: float):
        file_name = '/home/nadun/Documents/projects/flower/datasets/movie-conversation.csv'
        data = pd.read_csv(file_name)
        x_train, x_test, y_train, y_test = train_test_split(data['words'], data['character'], test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    def get_train_data(self):
        return self.train_dataset
    
    def get_test_data(self):
        return self.test_dataset
    
    def get_val_data(self):
        return self.val_dataset


if __name__ == '__main__':
    dataset = Data()

    print(dataset.get_train_data())