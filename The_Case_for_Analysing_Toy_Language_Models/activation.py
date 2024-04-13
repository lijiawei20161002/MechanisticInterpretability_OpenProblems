from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import GPT2Tokenizer
from functools import partial
import torch

# Define the configuration for the HookedTransformer
config = HookedTransformerConfig(
    n_layers=1,       # Number of transformer blocks (layers)
    d_model=768,       # Dimensionality of the model
    n_ctx=1024,        # Maximum sequence length
    d_head=64,         # Dimensionality of each attention head
    n_heads=12,        # Number of attention heads
    d_mlp=3072,        # Dimensionality of the feedforward network model
    act_fn='gelu',     # Activation function
    d_vocab=50257,     # Vocabulary size
    model_name='gpt2'  # Model name to load pretrained weights
)

# Instantiate the HookedTransformer with the defined config
model = HookedTransformer(config)

# Instantiate tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare input tokens
input_text = "Hello World"
tokens = tokenizer.encode(input_text, return_tensors='pt')

# Define a simple hook function to capture specific activations
def capture_specific_activation(module, input, output):
    print(f"Activation captured from module: {output}")
    return output  # It's essential to return the output, or it can alter the forward pass results.

# Register the hook for a specific layer and head (if the API permits)
if hasattr(model, 'layers'):
    specific_layer = model.layers[0]  # Example: hooking to the first layer
    specific_layer.register_forward_hook(capture_specific_activation)

# Run the model and print outputs
outputs = model(tokens)
print("Outputs from model with input: ", input_text, outputs)
