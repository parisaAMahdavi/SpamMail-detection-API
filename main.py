from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import re
import string
import os
from utils import load_tokenizer, set_seed
from data_loader import SpamEmailDataset, load_data
from trainer import train, evaluate, load_model
from app import GPT2Deployer
import transformers
import numpy as np
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args): 
    set_seed(args)
    tokenizer = load_tokenizer(args)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    train_data, dev_data, test_data = load_data(args, tokenizer)


    if args.do_train:
      model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=args.num_calss)
      model.resize_token_embeddings(len(tokenizer))
      model.config.pad_token_id = tokenizer.pad_token_id
      model.config.bos_token_id = tokenizer.bos_token_id
      model.to(device)
      gstep, gloss = train(args, model, train_data,dev_data, device, tokenizer)
      results = evaluate(args, mode='test', device=device, test_dataloader=test_data)

    elif args.do_test:
      test_results = evaluate(args, mode='test', device=device, test_dataloader=test_data)

    else:
      model = load_model(args, device)
      api = GPT2Deployer(model, tokenizer, args.max_seq_len)
      api.run_server()

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
    parser.add_argument("--do_train",  action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
