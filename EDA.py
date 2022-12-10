from transformers import BertConfig, BertModel
import torch

from torch import nn
import numpy as np
import pandas as pd
import pickle, time
import re, os, string, typing, gc, json
import torch.nn.functional as F
import spacy
from collections import Counter

from tqdm import tqdm
from transformers import BertTokenizer



from nltk.corpus import wordnet,stopwords
import random

from nltk.tokenize import sent_tokenize

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

def swap_word(new_words):
    
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        
        if counter > 3:
            return new_words
    
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def random_swap(words, n):
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        new_words = swap_word(new_words)
        
    sentence = ' '.join(new_words)
    
    return sentence

def add_word(new_words):
    
    synonyms = []
    counter = 0
    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
        
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def random_insertion(words, n):
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def random_deletion(words, n):

    words = words.split()
    
    #obviously, if there's only one word, don't delete it
    if len(words) == 1 or n>len(words):
        return words

    #randomly delete words with probability p
    L = random.sample(range(0, len(words)-1), n)
    new_words = [v for i,v in enumerate(words) if i not in L]

    sentence = ' '.join(new_words)
    
    return sentence

def Easy_Data_Augmentation(sent,case_num,n):
    #sent = tokenizer.decode([v for v in sent_vec if v not in [0,101,102]]) 
    
    # Synonym Replacement
    if case_num == 1:
        #sent_vec = tokenizer.encode(synonym_replacement(sent, n))
        sent_vec = synonym_replacement(sent, n)
    
    # Random Insertion
    if case_num == 2:
        #sent_vec = tokenizer.encode(random_insertion(sent, n))
        sent_vec = random_insertion(sent, n)
    
    # Random Swap
    if case_num == 3:
        #sent_vec = tokenizer.encode(random_swap(sent, n))
        sent_vec = random_swap(sent, n)
    
    # Random Deletion
    if case_num == 4:
        # sent_tok = tokenizer.encode(sent)
        # L = random.sample(range(1, len(sent_tok)-1), n)
        # sent_vec = [v for i,v in enumerate(sent_tok) if i not in L]
        sent_vec = random_deletion(sent, n)
    
    return sent_vec



























