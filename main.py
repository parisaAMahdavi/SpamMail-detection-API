# from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import re
import string
import os
from utils import load_tokenizer
from data_loader import SpamEmailDataset, load_data
from trainer import train, evaluate
import transformers

import numpy as np

# app = Flask(__name__)
def main(args): 

    tokenizer = load_tokenizer(args)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=args.num_calss)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    model.to(device)

    if args.do_train:
        train_data, dev_data, test_data = load_data(args, tokenizer, mode='train')
        gstep, gloss = train(args, model, train_data,dev_data, device, tokenizer)
        results = evaluate(args, mode='test', device=device, test_dataloader=test_data)

    if args.do_test:
        test_data = load_data(args, tokenizer, mode='test')
                            
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--data_dir", default="./data", type=str, help="the input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument('--num_calss', type=int, default=2, help="number of classes")
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=1e2, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_seq_len", default=200, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--model_name", default="gpt2", type= str , help="Name of transformers model - will use already pretrained model.")
    parser.add_argument("--cuda", action="store_false", help="using CUDA when available")
    # parser.add_argument("--mode", default='train', choices=['train', 'test', 'pred'], help="whether to run training or test on test_set or inference phase")
    parser.add_argument("--do_train",  action="store_false", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_false", help="Whether to run eval on the test set.")
    

    return parser.parse_args()





# def classify_text(text):
#     """
#     Classify the input text and return the predicted class.
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     predicted_class_id = logits.argmax().item()
#     return predicted_class_id


# @app.route('/classify', methods=['POST'])
# def classify():
#     data = request.json
#     text = data.get('text', '')

#     if not text:
#         return jsonify({'error': 'No text provided'}), 400

#     # Classify the text
#     predicted_class_id = classify_text(text)

#     # Return the classification result
#     return jsonify({'predicted_class': predicted_class_id})

if __name__ == '__main__':
    args = parse_args()
    main(args)
    # app.run(debug=True)
