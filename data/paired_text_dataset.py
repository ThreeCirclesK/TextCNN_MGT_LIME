from torch.utils.data import Dataset#, DataLoader
import torch

class PairedTextDataset(Dataset):
    def __init__(self, fidxs, data, vocab, tokenizer, max_length=128):
        self.fidxs = fidxs
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def encode(self, text):
        tokens = self.tokenizer(text)
        token_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens[:self.max_length]]  # Truncate
        return token_ids


    def __getitem__(self, idx):
        text1, text2, label = self.data[idx]
        tokens1 = torch.tensor(self.encode(text1), dtype=torch.long)
        tokens2 = torch.tensor(self.encode(text2), dtype=torch.long)
        fidx = self.fidxs[idx]
        return tokens1, tokens2, torch.tensor(label, dtype=torch.long), fidx