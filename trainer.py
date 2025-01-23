import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets, Dataset
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define the path to your text8 dataset
text8_path = 'text8.train.txt'  # path to  text8 dataset

# Load the tokenizer and model
model_name = "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load the Alpaca dataset
instruction_tuned_dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Tokenize the instruction tuned dataset
def tokenize_function(examples):
    # Combine instruction and input for tokenization
    combined_text = [i + j for i, j in zip(examples["instruction"], examples["input"])]
    tokenized_output = tokenizer(combined_text, truncation=True, padding="max_length", max_length=128)
    return tokenized_output

# Apply tokenization
instruction_tuned_dataset = instruction_tuned_dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output"])

# Convert instruction tuned dataset to a format suitable for training
instruction_tuned_dataset = Dataset.from_dict(instruction_tuned_dataset.to_dict())

# Load and tokenize text8 dataset
def load_text8_dataset(file_path, tokenizer, block_size=128):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    texts = [line.strip() for line in lines]
    tokenized_texts = tokenizer(texts, truncation=True, padding="max_length", max_length=block_size)
    dataset = Dataset.from_dict(tokenized_texts)
    return dataset

text8_dataset = load_text8_dataset(text8_path, tokenizer)

# Combine datasets
combined_dataset = concatenate_datasets([text8_dataset, instruction_tuned_dataset])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=200,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir='./logs',
    logging_steps=100
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=combined_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

# Function to plot training metrics
def plot_metrics(log_dir, plot_file):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    loss_values = []
    steps = []

    for event in event_acc.Scalars('loss'):
        steps.append(event.step)
        loss_values.append(event.value)

    plt.plot(steps, loss_values, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.savefig(plot_file)
    plt.close()

# Plot the training metrics and save to a file
plot_metrics(training_args.logging_dir, 'training_loss_plot.png')
