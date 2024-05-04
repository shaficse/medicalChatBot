import os
import torch

# Check if CUDA is available and set the device accordingly
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Maximum sentence length for processing
MAX_LENGTH = 50

# Minimum count threshold for trimming rare words from vocabulary
MIN_COUNT = 1

# Size of the hidden layers
hidden_size = 512

# Number of layers in the encoder and decoder GRU/LSTM
encoder_n_layers = 2
decoder_n_layers = 2

# Dropout rate to prevent overfitting
dropout = 0.1

# Batch size for training
batch_size = 32

# Training loop controls
clip = 50.0  # Clipping threshold to prevent exploding gradients
learning_rate_en = 0.0001  # Learning rate for the optimizer
learning_rate_de = 0.0005  # Learning rate for the optimizer
teacher_forcing_ratio = 0.5
decoder_learning_ratio = 5.0  # Ratio to adjust the decoder's learning rate relative to the encoder's
n_iteration = 15000  # Number of training iterations
print_every = 100  # Frequency of progress output
save_every = 1  # Frequency of saving the model

# Directory and corpus configurations
corpus_name = "MedQuad"
corpus = os.path.join("data", corpus_name)
save_dir = os.path.join("model")  # Directory to save model

# Model parameters and setup
model_name = 'cb_model'
attn_model = 'dot'  # Attention model type (e.g., 'dot', 'general', or 'concat')

# Decision to start training from scratch or load existing model
strt_scratch = True  
if strt_scratch:
    loadFilename = None
else:
    # Checkpoint for loading from a specific iteration
    loadFilename = os.path.join(save_dir, model_name, corpus_name, f'best_model_val_per.tar')
