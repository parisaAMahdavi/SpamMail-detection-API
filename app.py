import os
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from utils import cleaning_method, convert_data_to_features


class GPT2Deployer:

    def __init__(self, model, tokenizer, max_seq_len, device, label_dict):
        self.model = model
        self.app = Flask(__name__)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.label_dic = label_dict
        print(self.label_dic)

        self.app.route('/predict', methods=['POST'])(self.predict)
        self.app.route('/')(self.home)
    
    def home(self):
        return render_template('index.html')

    def predict(self):
        text = request.form['mail']
        if not text:
          return jsonify({'error': 'No text provided'}), 400

        cleaned_text = cleaning_method(text) 
        ids, mask = convert_data_to_features(cleaned_text, self.tokenizer, self.max_seq_len)
          
        self.model.eval()
        
        b_input_ids = torch.tensor([ids]).to(self.device)
        b_masks = torch.tensor([mask]).to(self.device)
        with torch.no_grad():
            outputs = self.model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                             labels=None)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predict_label = self.label_dic[predict_content[0]]

        return render_template('index.html', prediction_text='Mail successfully classified as {}'.format(predict_label)), 200

    def run_server(self):
        self.app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
