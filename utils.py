from transformers import GPT2Tokenizer
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import torch
import re
import random

def set_seed(args):
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

def get_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, 'labels.txt'), 'r', encoding='utf-8')]

def encode_labels(args):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(get_labels(args))
    dic_labels = dict(zip(encoded_labels, get_labels(args)))
    return encoded_labels, dic_labels

def cleaning_method(text):
    """text data preprocessing."""

    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Remove punctuation - can be adjusted depending on the need
    # text = re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    
    # Remove extra spaces and tabs
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def convert_data_to_features(text, tokenizer, max_sequence_len):

    encodings_dict = tokenizer(text, truncation=True, max_length=max_sequence_len, padding="max_length")
    return encodings_dict['input_ids'], encodings_dict['attention_mask']

def load_tokenizer(args):
    if args.do_train :
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, bos_token='<|startoftext|>', pad_token='<|pad|>')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    return tokenizer