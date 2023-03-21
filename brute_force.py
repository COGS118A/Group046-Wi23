from gpulsh import GPULSH
import numpy as np
import csv
import random
import string

WORD_LENGTH = 8

def ham_dist(w1,w2):
    return sum(ch1 != ch2 for ch1,ch2 in zip(w1,w2))

def min_ham_dist(w1, word_list):
    return min(ham_dist(w1,w2) for w2 in word_list)

# Read in English dictionary words (whose lengths are WORD_LENGTH)
with open('frequency-alpha-alldicts.txt') as f:
    next(f)
    words_gen = (line.split()[1].lower() for line in f)
    word_list = list(filter(lambda word: len(word) == WORD_LENGTH, words_gen))

print(min_ham_dist('nuoovyza', word_list))

#print(len(word_list))

# k = 1 # Number of hash functions... k=1 achieved 100% accuracy on a test set of 60,000 random strings, so good enough.
# gpu_lsh = GPULSH(k, word_list)

# # Generate mutated strings for dataset
# input_strings = ['nuoovyza']

# # Calculate the correct training/test labels for the dataset
# input_dists = gpu_lsh.minimum_hamming_distance_batch(input_strings,batch_size=2000)

# print(list(zip(input_strings, input_dists)))