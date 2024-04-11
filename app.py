import os
import torch
from flask import Flask, request, jsonify, render_template
from torch import masked
from transformers import pipeline
from utils import cleaning_method, convert_data_to_features


class GPT2Deployer:

    def __init__(self, model, tokenizer, max_seq_len):
        self.model = model
        self.app = Flask(__name__)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.app.route('/predict', methods=['POST'])(self.predict)
    
    def predict():
        data = request.get_json()
        text = data['text']

        if not text:
          return jsonify({'error': 'No text provided'}), 400

        cleaned_text = cleaning_method(text) 
        b_input_ids, b_masks = convert_data_to_features(cleaned_text, self.tokenizer, self.max_seq_len)
          
        self.model.eval()
        
        # b_input_ids = torch.tensor(id).to(device)
        # b_masks = torch.tensor(mask).to(device)
        
        with torch.no_grad():
            outputs = self.model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                             labels=None)
            _, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            predict_content = logits.argmax(axis=-1).flatten().tolist()

        return render_template('index.html', prediction_text='Mail successfully classified as {}'.format(predict_content)), 200

    def run_server(self):
        self.app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
