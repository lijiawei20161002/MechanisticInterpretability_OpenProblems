import torch
from transformers import GPT2Model, GPT2Config

# Initialize a simplified GPT-2 model configuration with a single layer
config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=1)
model = GPT2Model(config)

# Access the weights of the single transformer layer
transformer_layer = model.h[0]

# Access the weights for the query, key, value in self-attention
query_weights = transformer_layer.attn.c_attn.weight[:768]  # Assuming a split into query, key, value
key_weights = transformer_layer.attn.c_attn.weight[768:1536]
value_weights = transformer_layer.attn.c_attn.weight[1536:]

# For example, let's focus on the weights related to the first neuron in the query part
first_neuron_query_weights = query_weights[:, 0]

print("Weights of the first neuron in the query part of the attention mechanism:")
print(first_neuron_query_weights)
