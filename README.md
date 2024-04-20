# Email Classifier using GPT2 with REST API Deployment

This project focuses on fine-tuning the GPT-2 model to classify emails as spam or ham (non-spam). Additionally, it provides a REST API using PyTorch and Flask to deploy the model into a web interface, all containerized using Docker.

## Features

* Fine-tuned GPT-2 model for email classification.
* REST API built with Flask for deployment.
* Docker containerization for seamless deployment and scalability.

## Getting Started

1. Clone the repository.
2. cd repo
3. Install required packages (pip install -r requirements.txt)

### Training & Evaluation

`
python3 main.py --do_train --do_test --data_dir 'path/to/the/data' --model_dir 'path/to/the/model' 
`

### Usage
1. Start the Flask app:
`python3 main.py --do_pred --model_dir 'path/to/the/model'`

2. Send a POST request to the `/predict` endpoint with a piece of text as email content:

### Containerize

1. Build docker image
` docker build -t spam-mail-detection .`
2. Run the docker container
` docker container run --detach --publish 8000:8000 spam-mail-detection `