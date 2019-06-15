"""Data utilities."""
import torch

def read_word2vec_inText(file_path, device):
    with open(file_path, 'r') as f:
        head = f.readline().strip()
        word_num, emb_dim = [int(value) for value in head.split(' ')]
        word_to_idx = {}
        embedding = [0] * word_num
        for line in f:
            line = line.strip('\n\r')
            items = line.split(' ')
            word = items[0]
            vector = [float(value) for value in items[1:]]
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            embedding[idx] = vector
    embedding = torch.tensor(embedding, dtype=torch.float, device=device)
    return word_to_idx, embedding
