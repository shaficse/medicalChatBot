import unicodedata
import re
import os
import itertools
import torch
from io import open
import random


# Token definitions for special purposes in data handling
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token



class Voc:
    """Vocabulary mapping words to indexes for natural language processing tasks."""
    def __init__(self, name):
        """Initialize a new Vocabulary object.

        Args:
            name (str): The name of the vocabulary.
        """
        self.name = name
        self.trimmed = False
        self.word2index = {"<UNK>": UNK_token}
        self.word2count = {"<UNK>": 0}
        self.index2word = {UNK_token: "<UNK>", PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 4  # Count UNK, PAD, SOS, EOS

    def addSentence(self, sentence):
        """Add all words in a sentence to the vocabulary."""
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """Trim words below a certain count threshold.

        Args:
            min_count (int): The minimum count threshold.
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # Reinitialize dictionaries to keep only frequent words
        self.word2index = {"<UNK>": UNK_token}
        self.word2count = {"<UNK>": 0}
        self.index2word = {UNK_token: "<UNK>", PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 4  # Reset count to include only UNK, PAD, SOS, EOS initially

        for word in keep_words:
            self.addWord(word)


def read_conversation_from_file(file_path):
    """Read conversations from a file, splitting them into question-answer pairs.

    Args:
        file_path (str): The path to the conversation file.

    Returns:
        list of tuple: A list of question-answer tuples.
    """
    conversation = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split('\t')
            qa, ans = line[0], line[1]
            conversation.append((qa, ans))

    return conversation


def unicodeToAscii(s):
    """Convert a Unicode string to plain ASCII.

    Args:
        s (str): The string to convert.

    Returns:
        str: The ASCII converted string.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    """Normalize, lower, and clean non-letter characters from a string.

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name):
    """Read and parse a file to extract conversation pairs and initialize a vocabulary object.

    Args:
        datafile (str): The path to the data file.
        corpus_name (str): The name of the corpus.

    Returns:
        tuple: A vocabulary object and a list of normalized question-answer pairs.
    """
    print("Reading lines...")
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filterPair(p, MAX_LENGTH):
    """Check if both sentences in a pair are under the specified maximum length.

    Args:
        p (list): The pair of sentences.
        MAX_LENGTH (int): Maximum length a sentence can have.

    Returns:
        bool: True if both sentences are under the maximum length, False otherwise.
    """
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs, MAX_LENGTH):
    """Filter all pairs using the filterPair condition.

    Args:
        pairs (list of list): The list of sentence pairs to filter.

    Returns:
        list: A filtered list of sentence pairs.
    """
    return [pair for pair in pairs if filterPair(pair,MAX_LENGTH)]



def trimRareWords(voc, pairs, MIN_COUNT):
    """Trim rare words from the vocabulary and filter out pairs containing these words.

    Args:
        voc (Voc): The vocabulary object.
        pairs (list of list): The list of sentence pairs.
        MIN_COUNT (int): The minimum count threshold for words to keep.

    Returns:
        list: A list of sentence pairs that do not contain trimmed words.
    """
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence, output_sentence = pair
        keep_input = all(word in voc.word2index for word in input_sentence.split(' '))
        keep_output = all(word in voc.word2index for word in output_sentence.split(' '))
        if keep_input and keep_output:
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def split_data(pairs, train_percent=0.70, validation_percent=0.15, test_percent=0.15):
    """
    Splits the data into training, validation, and testing sets based on specified percentages.
    
    Args:
        pairs (list): A list of data pairs (e.g., sentence pairs) to be split.
        train_percent (float): The percentage of the total data to be used for training.
        validation_percent (float): The percentage of the total data to be used for validation.
        test_percent (float): The percentage of the total data to be used for testing.
    
    Returns:
        tuple: A tuple containing three lists of pairs corresponding to the training,
               validation, and testing datasets.
    """
    # Ensure the sum of percentages equals 1 (100%)
    if train_percent + validation_percent + test_percent != 1:
        raise ValueError("The sum of train, validation, and test percentages must equal 1.")
    
    # Shuffle the pairs to ensure random distribution of data across splits
    random.shuffle(pairs)
    
    # Calculate the number of elements for each dataset based on the specified percentages
    total_pairs = len(pairs)
    train_end = int(train_percent * total_pairs)
    validation_end = train_end + int(validation_percent * total_pairs)
    
    # Assign data to each dataset based on calculated indices
    train_pairs = pairs[:train_end]
    validation_pairs = pairs[train_end:validation_end]
    test_pairs = pairs[validation_end:]
    
    return train_pairs, validation_pairs, test_pairs
