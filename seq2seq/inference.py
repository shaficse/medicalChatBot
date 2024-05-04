from nltk.translate.bleu_score import sentence_bleu
import nltk
import torch
nltk.download('punkt')

def indexesFromSentence(voc, sentence, PAD_token, SOS_token, UNK_token, EOS_token, device):
    """Convert sentence to list of word indexes, adding SOS and EOS tokens.

    Args:
        voc (Voc): The vocabulary used for index lookup.
        sentence (str): The sentence to convert.

    Returns:
        list: A list of indexes representing the sentence.
    """
    return [SOS_token] + [voc.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token]

def evaluate(encoder, decoder, sentence, voc,PAD_token,SOS_token, UNK_token, EOS_token, device, max_length=10):
    ### Prepare input sentence
    # This requires that `indexesFromSentence` is a function that converts the input sentence to a list of indices
    indexes_batch = [indexesFromSentence(voc, sentence,PAD_token,SOS_token, UNK_token, EOS_token, device)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1).to(device)
    
    # Encoding
    encoder_outputs, encoder_hidden = encoder(input_batch, lengths)
    
    # Create initial decoder input (start with SOS_token)
    decoder_input = torch.tensor([[SOS_token]], device=device)  # Tensor for the SOS token
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # Initialize list to hold the decoded words
    decoded_words = []
    
    for _ in range(max_length):
        # Decoding step
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
        # Choose top word from output
        _, topi = decoder_output.topk(1)
        if topi.item() == EOS_token:
            break  # Stop the decoding process if the EOS token is reached
        else:
            decoded_words.append(voc.index2word[topi.item()])
        
        # Next input is the current output
        decoder_input = topi.squeeze().detach().view(1, -1)

    # Join words; omit 'SOS' if it is the first word in the sequence
    if decoded_words and decoded_words[0] == 'SOS':
        return ' '.join(decoded_words[1:])  # Skip 'SOS'
    return ' '.join(decoded_words)



def calculate_bleu_score(reference_texts, generated_text):
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
    generated_tokens = nltk.word_tokenize(generated_text.lower())
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score