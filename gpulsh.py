import numpy as np
import torch

class GPULSH:
    def __init__(self, k, word_list, device='cuda'):
        self.k = k
        self.word_list = word_list
        if len(self.word_list) == 0:
            return
        self.word_length = len(word_list[0])
        self.hash_tables = [{} for _ in range(k)]

        # Generate k random bit vectors for the random projections
        self.random_projections = torch.randint(0, 2, (k, self.word_length), device=device)

        # Convert words to numerical representation and calculate their hash values
        words_torch = torch.tensor([[ord(c) for c in word] for word in word_list], device=device)
        hash_values = torch.sum(torch.bitwise_xor(words_torch[:, None, :], self.random_projections) & 1, dim=-1).cpu().numpy()

        # Populate the hash tables with words and their corresponding hash values
        for word_idx, word in enumerate(word_list):
            for i in range(k):
                hash_val = hash_values[word_idx, i]
                if hash_val not in self.hash_tables[i]:
                    self.hash_tables[i][hash_val] = []
                self.hash_tables[i][hash_val].append(word)

    def minimum_hamming_distance_batch(self, input_strings, batch_size=500):
        if len(self.word_list) == 0:
            return []
        n_inputs = len(input_strings)
        min_distances = np.empty(n_inputs, dtype=int)

        # Process input strings in chunks to save time
        for i in range(0, n_inputs, batch_size):
            chunk_start = i
            chunk_end = min(i + batch_size, n_inputs)
            input_chunk = input_strings[chunk_start:chunk_end]
            min_distances[chunk_start:chunk_end] = self._minimum_hamming_distance_batch_chunk(input_chunk)

        return list(min_distances)

    def _minimum_hamming_distance_batch_chunk(self, input_strings):
        device = self.random_projections[0].device

        # Convert input strings to numerical representation
        input_torch = torch.tensor([[ord(c) for c in input_string] for input_string in input_strings], device=device)
        n_inputs = len(input_strings)
        min_distances = torch.full((n_inputs,), float("inf"), device=device)

        # Iterate through all k hash tables
        for i in range(self.k):
            hash_table_keys = list(self.hash_tables[i].keys())
            if not hash_table_keys:
                continue

            # Create a tensor with all candidates from the hash table
            candidates = torch.tensor(
                [[ord(c) for c in word] for key in hash_table_keys for word in self.hash_tables[i][key]],
                device=device,
            )

            # Calculate the Hamming distance matrix between input strings and candidates
            distance_matrix = torch.sum(input_torch[:, None, :] != candidates[None, :, :], dim=-1)

            # Find the index of the minimum distance candidate for each input string
            min_candidate_indices = torch.argmin(distance_matrix, dim=-1)

            # Update the minimum distances with the new minimum values
            min_distances = torch.min(
                min_distances, torch.gather(distance_matrix, 1, min_candidate_indices[:, None])[:, 0]
            )

        return min_distances.cpu().numpy()