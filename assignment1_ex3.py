from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample sentence
sentence = "Natural language processing is amazing, yet seems complicated!"

# Tokenize and encode
inputs = tokenizer(sentence, return_tensors='pt')

# Forward pass through the model
outputs = model(**inputs)

# Extract hidden states (last layer embeddings)
last_hidden_state = outputs.last_hidden_state

# Display token embeddings
print("Token embeddings shape:", last_hidden_state.shape)
print("Token embeddings:")
print(last_hidden_state)

# Optional: Convert token ids back to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("\nTokens:", tokens)

# Optional: Average pooling to get sentence-level embedding
sentence_embedding = last_hidden_state.mean(dim=1)
print("\nSentence embedding shape:", sentence_embedding.shape)
print("Sentence embedding:", sentence_embedding)
