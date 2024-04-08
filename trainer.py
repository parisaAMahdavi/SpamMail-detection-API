import os
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2ForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from utils import get_labels
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args, model, train_dataloader,val_dataloader, device, tokenizer):
    
    total_steps = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    #changing the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", total_steps)

    predictions_labels = []
    true_labels = []
    eval_results= []

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()

            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)
          
            outputs = model(b_input_ids,
                            labels=b_labels, 
                            attention_mask = b_masks,
                            token_type_ids=None
                           )
            loss, logits = outputs[:2]

            loss.backward()

            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % 3 == 0:
                eval_results.append(evaluate(args, mode="dev", model=model, device=device, val_dataloader=val_dataloader))

            logits = logits.detach().cpu().numpy()
            # Convert these logits to list of predicted labels values.
            true_labels += batch[2].numpy().flatten().tolist()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        
        evaluate(args, mode='dev', device=device, val_dataloader=val_dataloader, model=model)
    save_model(args, model, tokenizer)
    return global_step, tr_loss / global_step

def evaluate(args, mode, device, model, val_dataloader=None, test_dataloader=None):
    if mode == 'test':
        dataset = test_dataloader
        model = load_model(args, device)
    elif mode == 'dev':
        dataset = val_dataloader
    else:
        raise Exception("Only dev and test dataset available")

    # Eval!
    logger.info("***** Running evaluation on %s dataset *****", mode)
    logger.info("  Num examples = %d", len(dataset.dataset))
    total_eval_loss = 0.0
    nb_eval_steps = 0

    predictions_labels = []
    true_labels = []

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    for batch in tqdm(dataset, desc="Evaluating"):
        true_labels += batch[2].numpy().flatten().tolist()

        b_input_ids = batch[0].to(device)
        b_labels = batch[2].to(device)
        b_masks = batch[1].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
            eval_loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_eval_loss += eval_loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content
        nb_eval_steps += 1

    avg_eval_loss = total_eval_loss / len(dataset)
    results = {
            "val_loss": avg_eval_loss,
            "val_acc": accuracy_score(true_labels, predictions_labels)
        }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results
    
def save_model(args, model, tokenizer):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logger.info("Saving model checkpoint to %s", args.model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    # Save training arguments together with the trained model
    # torch.save(args, os.path.join(args.model_dir, 'training_args.bin'))

def load_model(args, device):
    if not os.path.exists(args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
    try:
        model = GPT2ForSequenceClassification.from_pretrained(args.model_dir)
        model.to(device)
        logger.info("***** Model Loaded *****")
        return model
    except:
        raise Exception("Some model files might be missing...")
