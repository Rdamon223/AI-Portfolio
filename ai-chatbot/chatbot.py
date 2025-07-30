import streamlit as st  # Imports Streamlit for app building.
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Loads model and tokenizer.
import torch  # For tensor operations.

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Loads tokenizer.
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Loads model.

tokenizer.pad_token = tokenizer.eos_token  # Sets pad token to EOS explicitly to avoid warning.

# GPU: If available, move model to GPU for faster generation using to_empty() to handle meta tensor issues.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Detects GPU or falls back to CPU.
if device != torch.device('meta'):  # Checks if model is on meta (default load); uses to_empty if moving.
    model.to_empty(device=device)
else:
    model.to(device)  # Fallback if already on meta.

def generate_response(prompt, max_length=50):  # Function takes user prompt and optional length limit.
    inputs = tokenizer.encode(prompt, return_tensors='pt', padding=True, truncation=True)  # Encodes prompt to tokens as PyTorch tensor.
    inputs = inputs.to(device)  # Moves inputs to device.
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()  # Creates attention mask: 1 for real tokens, 0 for padding.
    attention_mask = attention_mask.to(device)  # Moves mask to device.
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoids repeating phrases.
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,  # Greedy decoding for stability (no random sampling errors).
        repetition_penalty=1.2  # Penalizes repetition to make outputs more natural.
    )  # Generates text continuation.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decodes tokens back to readable text, skipping special chars.

st.title("Simple Chatbot")  # Sets app title.
prompt = st.text_input("You:")  # Creates input box for user prompt.
if st.button("Send"):  # Button triggers generation.
    if prompt:  # Checks if prompt is not empty to avoid errors.
        response = generate_response(prompt)  # Calls the function to generate response.
        st.write("Bot:", response)  # Displays bot response.
    else:
        st.write("Bot: Please enter a message!")