import torch
import streamlit as st
from torch.nn import functional as F
import functions

PATH = "aristotlegpt14M.pth"

device = torch.device('cpu')
model = functions.AristotleGPTModel()
model.load_state_dict(torch.load(PATH, map_location=device))

with open('aristotle_texts.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#context = torch.tensor(encode("It is obvious that")).unsqueeze(0)
#print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

context_window = 256
def generate_stream(model_instance, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # Assume context_window is accessible or use model_instance.context_window if it's an attribute
        idx_cond = idx[:, -context_window:]
        logits, _ = model_instance(idx_cond)  # Assuming the model instance has a callable method for predictions
        logits = logits[:, -1, :]  # Focus only on the last time step
        probs = F.softmax(logits, dim=-1)  # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the distribution
        idx = torch.cat((idx, idx_next), dim=1)  # Append sampled index to the running sequence
        yield decode(idx_next[0].tolist())

# Streamlit app layout
st.title('AristotleGPT')

# User input
user_input = st.text_input("Enter text to generate from:", "There is a difference between")

# Generate button
if st.button('Generate'):
    context = torch.tensor(encode(user_input)).unsqueeze(0)
    st.write_stream(generate_stream(model, context, max_new_tokens=400))
    #st.write_stream(generated_text)