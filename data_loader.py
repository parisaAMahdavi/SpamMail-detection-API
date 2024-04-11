from torch.utils.data import Dataset, DataLoader, random_split
import os
import logging
import torch
from utils import cleaning_method, convert_data_to_features, get_labels
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
        self.labels = get_labels(args)
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
        self._encode_labels()
        self._encode_samples()

    def _encode_labels(self):
        self.labels = self.label_encoder.fit_transform(self.labels)
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

def load_data(args, tokenizer):

    dataset = SpamEmailDataset(args, tokenizer=tokenizer)
    train_size = int(0.7*len(dataset))
    val_size = int(0.15*len(dataset))
    test_size =  len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])         

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


