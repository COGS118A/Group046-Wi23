from gpulsh import GPULSH
import numpy as np
import csv
import random
import string

N = 5000000
MAX_LENGTH = 8

# Used to build dataset of near-words (some of which will be actual words)
# Uniformly chooses between 0 and WORD_LENGTH characters to randomly change in the word
def mutate_word(word):
    mutation_count = random.randint(0, len(word))
    mutated_indices = random.sample(range(len(word)), mutation_count)
    mutated_chars = [random.choice(string.ascii_lowercase) for _ in range(mutation_count)]
    mutated_word = list(word)
    for index, char in zip(mutated_indices, mutated_chars):
        mutated_word[index] = char
    return ''.join(mutated_word)

def pad_strings(words):
    return [word.ljust(MAX_LENGTH, '_') for word in words]


word_lists = [[] for _ in range(MAX_LENGTH)]
# Read in English dictionary words (whose lengths are WORD_LENGTH)
with open('frequency-alpha-alldicts.txt') as f:
    next(f)
    for line in f:
        word = line.split()[1].lower()
        if 1 <= len(word) <= MAX_LENGTH:
            word_lists[len(word)-1].append(word)

for i in range(MAX_LENGTH):
    print(len(word_lists[i]))

k = 1
gpu_lshs = [GPULSH(k, word_lists[i]) for i in range(MAX_LENGTH)]


total_words = sum([len(word_lists[i]) for i in range(MAX_LENGTH)])
props = [len(word_lists[i]) / total_words for i in range(MAX_LENGTH)]
sizes = [round(N*props[i]) for i in range(MAX_LENGTH)]

# Generate mutated strings for dataset
input_strings_arr = [[mutate_word(random.choice(word_lists[i])) for _ in range(sizes[i])] for i in range(MAX_LENGTH)]

# Calculate the correct training/test labels for the dataset
input_dists_arr = [gpu_lshs[i].minimum_hamming_distance_batch(input_strings_arr[i],batch_size=2000) for i in range(MAX_LENGTH)]

input_strings = pad_strings([input_string for input_strings in input_strings_arr for input_string in input_strings])
input_dists = [input_dist for input_dists in input_dists_arr for input_dist in input_dists]
data = list(zip(input_strings, input_dists))
random.shuffle(data)

# Write the generated data to a csv file
file_name = 'mixed_data' + str(MAX_LENGTH) + '.csv'
with open(file_name, mode='w', newline='') as f:    # newline = '' is a Windows compatibility thing
    writer = csv.writer(f)
    writer.writerows(data)