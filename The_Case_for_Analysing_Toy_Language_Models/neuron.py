import torch
from transformers import GPT2Model, GPT2Config

d_model = 768  
num_heads = 12
head_dim = d_model // num_heads  

# Initialize a simplified GPT-2 model configuration with a single layer
config = GPT2Config(
    n_layers=1,       # Number of transformer blocks (layers)
    d_model=d_model,       # Dimensionality of the model
    n_ctx=1024,        # Maximum sequence length
    d_head=64,         # Dimensionality of each attention head
    n_heads=num_heads,        # Number of attention heads
    d_mlp=3072,        # Dimensionality of the feedforward network model
    act_fn='gelu',     # Activation function
    d_vocab=50257,     # Vocabulary size
    model_name='gpt2'  # Model name to load pretrained weights
)
model = GPT2Model(config)

# Access the weights of the single transformer layer
transformer_layer = model.h[0]

# Access the weights for the query, key, value in self-attention
query_weights = transformer_layer.attn.c_attn.weight[:, :d_model]  
key_weights = transformer_layer.attn.c_attn.weight[:, d_model:2*d_model]
value_weights = transformer_layer.attn.c_attn.weight[:, d_model*2:]

neuron_index = 448
neuron_query_weights = query_weights[:, neuron_index]
neuron_key_weights = key_weights[:, neuron_index]
neuron_value_weights = value_weights[:, neuron_index]

print(f'Weights of the {neuron_index} neuron in the query part of the attention mechanism:')
print("Query Weight:", neuron_query_weights)
print("Key Weight:", neuron_key_weights)
print("Value Weight:", neuron_value_weights)
