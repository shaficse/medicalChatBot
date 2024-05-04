import math
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
import pickle

def zeroPadding(l, fillvalue):
    """Pad a list of lists with a fill value.

    Args:
        l (list of list): List of sequences.
        fillvalue (int, optional): The value to use for padding.

    Returns:
        list: A list of sequences padded to the length of the longest sequence.
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value):
    """Create a binary matrix indicating the presence of values other than the pad token.

    Args:
        l (list of list): List of padded sequences.
        value (int): The pad value to check against.

    Returns:
        list: A binary matrix corresponding to the input sequences.
    """
    m = []
    for seq in l:
        m.append([1 if token != value else 0 for token in seq])
    return m

def inputVar(l, voc, PAD_token,SOS_token, UNK_token, EOS_token, device):
    """Prepare the input variable by converting each sentence in the list to indexes and padding them.

    Args:
        l (list of str): List of sentences.
        voc (Voc): The vocabulary object used for word to index conversion.

    Returns:
        tuple: A tensor of padded input sequences and a tensor of their corresponding lengths.
    """
    indexes_batch = [indexesFromSentence(voc, sentence,PAD_token,SOS_token, UNK_token, EOS_token, device) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch,PAD_token)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def indexesFromSentence(voc, sentence, PAD_token, SOS_token, UNK_token, EOS_token, device):
    """Convert sentence to list of word indexes, adding SOS and EOS tokens.

    Args:
        voc (Voc): The vocabulary used for index lookup.
        sentence (str): The sentence to convert.

    Returns:
        list: A list of indexes representing the sentence.
    """
    return [SOS_token] + [voc.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token]

def outputVar(l, voc,PAD_token, SOS_token, UNK_token, EOS_token, device):
    """Prepare the output variable for the model, including a mask for the padding and the max target length.

    Args:
        l (list of str): List of output sentences.
        voc (Voc): The vocabulary object used for word to index conversion.

    Returns:
        tuple: A tensor of padded target sequences, a binary mask tensor indicating non-pad elements, and an integer for the maximum target length.
    """
    indexes_batch = [indexesFromSentence(voc, sentence, PAD_token, SOS_token, UNK_token, EOS_token, device) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, PAD_token)
    mask = binaryMatrix(padList, PAD_token)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch,PAD_token,SOS_token, UNK_token, EOS_token, device):
    """Organize training data into batches for training. Sorts batches by the lengths of sentences in descending order.

    Args:
        voc (Voc): The vocabulary object used for indexing.
        pair_batch (list of tuples): Batch of (input, output) pairs.

    Returns:
        tuple: Tensors of inputs, input lengths, outputs, output masks, and the maximum output length.
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc,PAD_token,SOS_token, UNK_token, EOS_token, device)
    output, mask, max_target_len = outputVar(output_batch, voc, PAD_token, SOS_token, UNK_token, EOS_token, device)
    return inp, lengths, output, mask, max_target_len


def maskNLLLoss(inp, target, mask, device):
    """
    Calculate the masked negative log likelihood loss.

    Args:
        inp (torch.Tensor): The prediction log probabilities from the model.
        target (torch.Tensor): The ground truth label indices.
        mask (torch.Tensor): Boolean tensor corresponding to target indicating which elements should be included in loss calculation.
        device (torch.device): The device tensors are on.

    Returns:
        tuple: A tensor representing the loss and the number of elements considered in the loss calculation.
    """
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()



def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, batch_size, clip, teacher_forcing_ratio, SOS_token, PAD_token, UNK_token, EOS_token, device, train_mode=True):
    
    """
    Perform a single training step including forward and backward pass and parameters update.

    Args:
        input_variable, lengths, target_variable, mask, max_target_len (torch.Tensor): Inputs and targets for the training step.
        encoder, decoder (nn.Module): Encoder and decoder models.
        encoder_optimizer, decoder_optimizer (torch.optim.Optimizer): Optimizers for the encoder and decoder.
        batch_size (int): Number of samples in a single batch.
        clip (float): Gradient clipping threshold.
        teacher_forcing_ratio (float): Probability to use teacher forcing during training.
        {SOS_token, PAD_token, UNK_token, EOS_token} (int): Special tokens used in data processing.
        device (torch.device): Device on which to perform computations.
        train_mode (bool): Whether the function is being called in training mode or not.

    Returns:
        tuple: Average loss and perplexity for the batch.
    """
    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    lengths = lengths.to("cpu")  # Lengths for RNN packing should always be on the CPU

    if train_mode:
        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    if train_mode:
        # Perform backpropagation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

    return sum(print_losses) / n_totals, math.exp(sum(print_losses) / n_totals)


model_metrices = []

# Modify trainIters to track and save based on perplexity
def trainIters(model_name, voc, training_pairs, validation_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, PAD_token, SOS_token, UNK_token, EOS_token,device,teacher_forcing_ratio):
    """
    Run multiple iterations of training and validation, managing model saving and logging progress.

    Args:
        model_name (str): Name of the model for saving purposes.
        voc (Voc): Vocabulary object.
        training_pairs (list): List of training data pairs.
        validation_pairs (list): List of validation data pairs.
        encoder, decoder (nn.Module): Encoder and decoder neural network models.
        encoder_optimizer, decoder_optimizer (torch.optim.Optimizer): Optimizers for the encoder and decoder.
        embedding (nn.Embedding): Embedding layer used in the encoder and decoder.
        encoder_n_layers, decoder_n_layers (int): Number of layers in the encoder and decoder.
        save_dir (str): Directory to save model checkpoints.
        n_iteration (int): Number of training iterations to perform.
        batch_size (int): Number of pairs per batch.
        print_every (int): Frequency of printing training progress.
        save_every (int): Frequency of saving the current model.
        clip (float): Maximum gradient norm.
        corpus_name (str): Name of the dataset or corpus.
        loadFilename (str): Path to a pre-trained model checkpoint file.
        {PAD_token, SOS_token, UNK_token, EOS_token} (int): Special tokens used in data processing.
        device (torch.device): Device configuration.
        teacher_forcing_ratio (float): Frequency of using teacher forcing during training.

    Returns:
        None: This function does not return but prints training and validation statistics.
    """
    # Prepare training and validation batches
    training_batches = [batch2TrainData(voc, [random.choice(training_pairs) for _ in range(batch_size)],PAD_token, SOS_token, UNK_token, EOS_token, device)
                        for _ in range(n_iteration)]
    validation_batches = [batch2TrainData(voc, [random.choice(validation_pairs) for _ in range(batch_size)],PAD_token, SOS_token, UNK_token, EOS_token, device)
                          for _ in range(1)]  

    # Initialization
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    best_perplexity = float('inf')
    if loadFilename:
        checkpoint = torch.load(loadFilename, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        start_iteration = checkpoint['iteration'] + 1
        print_loss = checkpoint['loss']
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        if 'embedding' in checkpoint:
            embedding.load_state_dict(checkpoint['embedding'])
        voc.__dict__ = checkpoint['voc_dict']
        print(f"Resuming training from iteration {start_iteration}")

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Training step
        loss, train_perplexity = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip,  teacher_forcing_ratio, PAD_token, SOS_token, UNK_token, EOS_token, device,train_mode=True)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(f"Iteration: {iteration}; Percent complete: {iteration / n_iteration * 100:.1f}%; Training loss: {print_loss_avg:.4f}; Perplexity: {train_perplexity:.4f}")
            print_loss = 0

            # Validation step
            validation_loss, validation_perplexity = validate(encoder, decoder, validation_batches, batch_size,PAD_token, SOS_token, UNK_token, EOS_token, device)
            print(f"Validation Loss: {validation_loss:.4f}; Validation Perplexity: {validation_perplexity:.4f}")

            # Save based on best perplexity
            if validation_perplexity < best_perplexity:
                best_perplexity = validation_perplexity
                directory = os.path.join(save_dir, model_name, corpus_name)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': validation_loss,
                    'perplexity': validation_perplexity,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, f'best_model_val_per.tar'))
            # # Save based on best perplexity
            # if train_perplexity < best_perplexity:
            #     best_perplexity = train_perplexity
            #     directory = os.path.join(save_dir, model_name, corpus_name)
            #     if not os.path.exists(directory):
            #         os.makedirs(directory)
                
            #     torch.save({
            #         'iteration': iteration,
            #         'en': encoder.state_dict(),
            #         'de': decoder.state_dict(),
            #         'en_opt': encoder_optimizer.state_dict(),
            #         'de_opt': decoder_optimizer.state_dict(),
            #         'loss': loss,
            #         'perplexity': train_perplexity,
            #         'voc_dict': voc.__dict__,
            #         'embedding': embedding.state_dict()
            #     }, os.path.join(directory, f'best_model_perplexity.tar'))

            model_metrices.append({'Iteration':iteration,  'Training loss': loss, "Training Perplexity":train_perplexity,  'Validation loss': validation_loss, "Validation Perplexity":validation_perplexity})
    return model_metrices
            

# Modified validate function to compute and return validation perplexity
def validate(encoder, decoder, validation_batches, batch_size,PAD_token, SOS_token, UNK_token, EOS_token, device):
    """
    Validate the model performance on a validation set.

    Args:
        encoder, decoder (nn.Module): Encoder and decoder models.
        validation_batches (list of tuples): Preprocessed batches of validation data.
        batch_size (int): Number of samples per batch.
        {PAD_token, SOS_token, UNK_token, EOS_token} (int): Special tokens used in data processing.
        device (torch.device): Device configuration.

    Returns:
        tuple: Average validation loss and perplexity.
    """
    with torch.no_grad():  # No need to track gradients
        validation_loss = 0
        for i, batch in enumerate(validation_batches):
            input_variable, lengths, target_variable, mask, max_target_len = batch
            input_variable = input_variable.to(device)
            target_variable = target_variable.to(device)
            mask = mask.to(device)
            lengths = lengths.to("cpu")  # Ensure lengths are on CPU
            
            loss, _ = train(input_variable, lengths, target_variable, mask, max_target_len,
                        encoder, decoder, None, None, batch_size, 0, 0,PAD_token, SOS_token, UNK_token, EOS_token, device, train_mode=False)
            validation_loss += loss
        
        average_validation_loss = validation_loss / len(validation_batches)
        validation_perplexity = math.exp(average_validation_loss)
    return average_validation_loss, validation_perplexity

