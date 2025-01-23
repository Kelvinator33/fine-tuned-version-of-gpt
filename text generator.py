import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Load the fine-tuned model and tokenizer
model_name = "drive/MyDrive/results"  # Directory where the model is saved
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set up the text generation pipeline
chatbot_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Function to generate a response from the chatbot
def get_response(prompt):
    response = chatbot_pipeline(prompt, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return response[0]['generated_text']

# Interactive loop
print("Chatbot is ready! Type your questions below. Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print(f"Chatbot: {response[len(user_input):].strip()}")
