import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def get_dataset():
    base_path = '/home/nadun/Documents/projects/flower/simulation_env/data/stack_overflow/'
    train = pd.read_csv(f'{base_path}/train.csv')[['label', 'text']]
    test = pd.read_csv(f'{base_path}/test.csv')[['label', 'text']]
    val = pd.read_csv(f'{base_path}/val.csv')[['label', 'text']]

    # preprocess
    rows = []
    for i, row in train.iterrows():
        r = preprocess_data(row)
        rows.append(r)

    train = pd.DataFrame(rows)

    return train, test, val


def preprocess_data(row):
    text = row['text']

    words = text.split()
    train_d, test_d = train_test_split(words, test_size=0.2)

    vocab = set(words)
    word_to_index = { word: i for i, word in enumerate(vocab) }

    indices1 = [word_to_index[word] for word in train_d]
    indices2 = [word_to_index[word] for word in test_d]

    train_data = torch.tensor(indices1 + indices2, dtype=torch.long).view(1, -1)
    train_targets = torch.tensor(indices1 + indices2, dtype=torch.long).view(1, -1)

    test_data = torch.tensor(indices2, dtype=torch.long).view(1, -1)  # Test sequence
    test_targets = torch.tensor(indices2, dtype=torch.long).view(1, -1)  # Test targets

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_targets), batch_size=1)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_targets), batch_size=1)

    data = {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'user_id': row['label'],
        'vocab_size': len(vocab)
    }
    return data