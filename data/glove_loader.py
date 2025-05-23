import torch

def load_glove(embedding_dim=100):
    vocab = {"<pad>": 0, "<unk>": 1}
    vectors = [torch.zeros(embedding_dim), torch.randn(embedding_dim)]  # pad & unk

    with open(f"artifacts/glove6B/glove.6B.{embedding_dim}d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            if len(vec) == embedding_dim:
                vocab[word] = len(vocab)
                vectors.append(vec)
    
    embedding_matrix = torch.stack(vectors)
    return vocab, embedding_matrix
