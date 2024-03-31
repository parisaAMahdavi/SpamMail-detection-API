from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
from utils import cleaning_method, convert_data_to_features
from sklearn.preprocessing import LabelEncoder


class SpamEmailDataset(Dataset):
    """
    data parsing happens within this class.

    Args:
        data_path (string): Directory with the data files.
        transform (callable): some cleaninig and preprocessing steps applied on each sample.
    """
    def __init__(self, args, tokenizer):

        self.data_path = args.data_dir
        self.input_text_file = 'mails.txt'
        self.label_file = 'labels.txt'
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.label_encoder = LabelEncoder()
        self.input_ids = []
        self.attn_masks = []

        data_file = os.path.join(self.data_path, self.input_text_file)
        labels_file = os.path.join(self.data_path, self.label_file)

        with open(data_file, 'r') as df:
            self.data = []
            for line in df:
                self.data.append(line.strip())
        
        with open(labels_file, 'r') as lf:
            self.labels = []
            for label in lf:
                self.labels.append(label.strip())
        
        self._encode_labels()

    def _encode_labels(self):
        self.labels = self.label_encoder.fit_transform(self.labels)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        sample = self.data[idx]
        label = self.labels[idx]
        
        sample = cleaning_method(sample)
        
        features = convert_data_to_features(sample, self.tokenizer, self.max_seq_len)
        self.input_ids.append(features['input_ids'])
        self.attn_masks.append(features['attn_masks'])

        label_tensor = torch.tensor(label, dtype=torch.long)
        return self.input_ids, self.attn_masks, label_tensor


def load_data(args, tokenizer, mode):

    dataset = SpamEmailDataset(args, tokenizer=tokenizer)
    train_size = int(0.7*len(dataset))
    val_size = int(0.15*len(dataset))
    test_size =  len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])         

    if mode == 'train':
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        return train_dataloader, val_dataloader
    elif mode == "test":
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return test_dataloader

