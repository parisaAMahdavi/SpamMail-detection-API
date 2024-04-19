FROM python:3.9-slim

# working directory in the container
WORKDIR /predict

# Copy the current directory contents into the container at /app
COPY . /predict

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
# Train and test before prediction
# RUN pip3 main.py --do_train --do_test --data_dir ./data --model_dir ./model
# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run main.py when the container launches
CMD ["python3", "main.py", "--do_pred", "--model_dir", "./model"]
