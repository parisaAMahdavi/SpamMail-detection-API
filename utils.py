from transformers import GPT2Tokenizer
import os
import numpy as np
import torch
import re
import string


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
    text = re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    
    # Remove extra spaces and tabs
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def convert_data_to_features(text, tokenizer, max_sequence_len):

    encodings_dict = tokenizer('<|startoftext|>'+ text + '<|endoftext|>', return_tensors="pt", truncation=True, max_length=max_sequence_len, padding="max_length")

    features = {'input_ids':torch.tensor(encodings_dict['input_ids']), 
                       'attn_masks': torch.tensor(encodings_dict['attention_mask'])}

    return features

def load_tokenizer(model_name):
    return GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>', pad_token='<|pad|>')

