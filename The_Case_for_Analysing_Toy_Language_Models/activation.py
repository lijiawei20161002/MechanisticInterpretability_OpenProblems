from transformers import GPT2Tokenizer
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
import numpy as np

# Define your configuration
config = HookedTransformerConfig(
    n_layers=1,  
    d_model=768, 
    d_head=64, 
    n_heads=12, 
    d_mlp=3072,
    act_fn='gelu', 
    n_ctx=1024, 
    d_vocab=50257,
    model_name='gpt2'
)

# Instantiate model
model = HookedTransformer(config)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example texts
texts = ["Hello world", "Example of input text", "Another text input for model"]

# Store activations globally
activations = []

# Define a hook function to capture and store activations
def capture_activations(module, input, output):
    # Average across the sequence length (token) dimension to get consistent shapes
    averaged_activation = output.detach().mean(dim=1).cpu().numpy()
    activations.append(averaged_activation)

# Attach hook to the appropriate layers
if hasattr(model, 'blocks'):
    for block in model.blocks:
        # Attach hook based on your model's specific layers or sub-layers
        if hasattr(block, 'attn'):
            block.attn.register_forward_hook(capture_activations)
        elif hasattr(block, 'mlp'):
            block.mlp.register_forward_hook(capture_activations)

# Run model with hooks
for text in texts:
    tokens = tokenizer.encode(text, return_tensors='pt')
    model(tokens)

# Concatenate all captured activations
all_activations = np.concatenate(activations, axis=0)  
max_activation_index = np.argmax(np.mean(all_activations, axis=0))
print("Neuron with the highest average activation:", max_activation_index)
