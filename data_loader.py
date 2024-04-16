from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
import random
import logging
import torch
from utils import cleaning_method, convert_data_to_features, encode_labels
from sklearn.preprocessing import LabelEncoder
logger = logging.getLogger(__name__)


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
        self.labels = encode_labels(args)[0]
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.label_encoder = LabelEncoder()
        self.input_ids = []
        self.attn_masks = []

        data_file = os.path.join(self.data_path, self.input_text_file)

        with open(data_file, 'r') as df:
            self.data = []
            for line in df:
                self.data.append(line.strip())   

        self.data = self.data
        self._encode_samples()

    def _encode_samples(self):
        for i,sample in enumerate(self.data):
          cleaned = cleaning_method(sample)        
          id, mask = convert_data_to_features(cleaned, self.tokenizer, self.max_seq_len)
          self.input_ids.append(id)
          self.attn_masks.append(mask)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):

        id_tensor = torch.tensor(self.input_ids[idx])
        mask_tensor = torch.tensor(self.attn_masks[idx])
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return id_tensor, mask_tensor, label_tensor

def load_data(args, dataset):

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Calculate sizes
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))
    test_size = len(indices) - train_size - val_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoader with the specified samplers
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


