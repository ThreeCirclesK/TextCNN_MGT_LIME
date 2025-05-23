from lime.lime_text import LimeTextExplainer
import torch
from torch.nn.utils.rnn import pad_sequence
from data import pad_to_min_length
import numpy as np


class LimeDualWrapper:
    def __init__(self, model, vocab, tokenizer, fixed_text, device, analyze='input1', min_length=5, batch_size=64):
        assert analyze in ['input1', 'input2']
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.device = device
        self.analyze = analyze
        self.min_length = min_length
        self.batch_size = batch_size

        # Tokenize and pad the fixed input
        fixed_tokens = [vocab.get(tok, vocab["<unk>"]) for tok in tokenizer(fixed_text)]
        if len(fixed_tokens) < min_length:
            fixed_tokens += [vocab["<pad>"]] * (min_length - len(fixed_tokens))
        self.fixed_tensor = torch.tensor(fixed_tokens, dtype=torch.long).unsqueeze(0).to(device)

    def predict(self, texts):
        all_probs = []
        batch_size = self.batch_size  # Make sure this is set in __init__
    
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
    
            perturbed_tokens = []
            for text in batch_texts:
                tokens = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in self.tokenizer(text)]
                if len(tokens) < self.min_length:
                    tokens += [self.vocab["<pad>"]] * (self.min_length - len(tokens))
                perturbed_tokens.append(torch.tensor(tokens, dtype=torch.long))
    
            padded_perturbed = pad_sequence(
                perturbed_tokens, batch_first=True, padding_value=self.vocab["<pad>"]
            ).to(self.device)
    
            # Apply minimum sequence padding globally (e.g., to 5 tokens)
            padded_perturbed = pad_to_min_length(padded_perturbed, self.vocab["<pad>"], 5)
    
            # Pad fixed input to match current batch's sequence length
            fixed_tokens = self.fixed_tensor.squeeze(0).tolist()
            seq_len = padded_perturbed.size(1)
            if len(fixed_tokens) < seq_len:
                fixed_tokens += [self.vocab["<pad>"]] * (seq_len - len(fixed_tokens))
            else:
                fixed_tokens = fixed_tokens[:seq_len]
            fixed_batch = torch.tensor(fixed_tokens, dtype=torch.long).unsqueeze(0).expand(padded_perturbed.size(0), -1).to(self.device)
    
            if self.analyze == 'input1':
                input1_batch = padded_perturbed
                input2_batch = fixed_batch
            else:
                input1_batch = fixed_batch
                input2_batch = padded_perturbed
    
            with torch.no_grad():
                logits = self.model(input1_batch, input2_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
    
            # Cleanup
            del logits, perturbed_tokens, input1_batch, input2_batch, padded_perturbed, fixed_batch
            torch.cuda.empty_cache()
    
        return np.vstack(all_probs)

