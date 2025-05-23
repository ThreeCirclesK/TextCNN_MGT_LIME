import os
import pickle
import json
from sklearn.model_selection import train_test_split
import re
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from data import (
    load_glove,
    PairedTextDataset,
)

def simple_tokenizer(text):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in STOPWORDS]

def encode(text, vocab, tokenizer, max_length):
    tokens = tokenizer(text)
    token_ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens[:max_length]]  # Truncate
    return token_ids

# === Function to generate paired_data from bugs ===
def build_paired_data(bug_list, indexing_dir, brid2commit):
    data = []
    fidx_list = []
    positive_count = 0

    for bug in bug_list:
        brid = bug['number']
        fixed_files = bug['fixed']
        
        if 'title' in bug:  # Train & Validation Dataset
            title = bug['title']
            body = bug['body']
            brtext = f"{title}\n{body}"
            commit_sha = brid2commit[brid]
        else:  # Test Dataset
            brtext = bug['brtext']
            commit_sha = bug['parent_commit']

        index_filename = f"{brid}_{commit_sha}.json"
        index_path = os.path.join(indexing_dir, index_filename)

        if not os.path.exists(index_path):
            continue  # skip missing index

        idx2file, _, file2idx = read_index_file(index_path)

        for idx, filename in idx2file.items():
            label = 1 if filename in fixed_files else 0
            if label == 1:
                positive_count += 1

            file_path = os.path.join(indexing_dir, idx)
            try:
                with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
                    sftext = f.read()
                data.append((brtext, sftext, label))
                fidx_list.append(idx)
            except FileNotFoundError:
                continue

    # === Compute class weights ===
    total = len(data)
    negative_count = total - positive_count
    class_counts = [negative_count, positive_count]
    weights = [total / c if c > 0 else 0.0 for c in class_counts]

    return data, weights, fidx_list

# === Function to generate paired_data from bugs, removing mgts===
def build_paired_data_wmgt(bug_list, indexing_dir, mgts):
    data = []
    fidx = []
    positive_count = 0

    for bug in bug_list:
        brid = bug['number']
        fixed_files = bug['fixed']
        
        if 'title' in bug.keys(): ## Train & Validation Dataset
            title = bug['title']
            body = bug['body']
            brtext = f"{title}\n{body}"
            commit_sha = brid2commit[brid]
        else: ## Test Dataset
            brtext = bug['brtext']
            for mgt in mgts:
                brtext = brtext.replace(mgt, '')
            commit_sha = bug['parent_commit'] 

        index_filename = f"{brid}_{commit_sha}.json"
        index_path = os.path.join(indexing_dir, index_filename)
        idx2file, _, file2idx = read_index_file(index_path)

        for idx, filename in idx2file.items():
            label = 1 if filename in fixed_files else 0
            if label == 1:
                positive_count += 1

            file_path = os.path.join(indexing_dir, idx)
            with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
                sftext = f.read()
                for mgt in mgts:
                    sftext = sftext.replace(mgt, '')
            data.append((brtext, sftext, label))
            fidx.append(idx)

    # === Compute class weights ===
    total = len(data)
    negative_count = total - positive_count 
    class_counts = [negative_count, positive_count] #class 0 = majority, class 1 = minority

    # Avoid division by zero
    weights = [total / c if c > 0 else 0.0 for c in class_counts]

    return data, weights, fidx
    


# === Function to generate train/test paired data ===
def load_and_prepare_bug_data(reponame: str, repodir: str, split_ratio: float = 0.33):
    repo_path = f"{repodir}/{reponame}"
    indexing_dir = os.path.abspath(f"{repo_path}_indexing")

    with open(f"datasets/past_brid2commit/{reponame}_bugs_brid2commit.pkl", 'rb') as f:
        bugs, brid2commit = pickle.load(f)
        print(f"{len(bugs)} past BRs, {len(brid2commit)} BRs with mapped commits")
        bugs = [x for x in bugs if x['number'] in brid2commit]

    train_bugs, val_bugs = train_test_split(bugs, test_size=split_ratio, random_state=42)
    train_paired_data, train_weights, tfidxs = build_paired_data(train_bugs, indexing_dir, brid2commit)
    val_paired_data, val_weights, vfidxs = build_paired_data(val_bugs, indexing_dir, brid2commit)

    return tfidxs, vfidxs, train_paired_data, train_weights, val_paired_data, val_weights

# === Function to build vocab and DataLoaders ===
def prepare_dataloaders(tfidxs, vfidxs, train_paired_data, val_paired_data, batch_size=32, embedding_dim=100):
    vocab, embedding_matrix = load_glove(embedding_dim=embedding_dim)
    tokenizer = simple_tokenizer
    _, max_len = compute_length_percentiles(train_paired_data, tokenizer)

    train_dataset = PairedTextDataset(tfidxs, train_paired_data, vocab, tokenizer, max_length=max_len)
    val_dataset = PairedTextDataset(vfidxs, val_paired_data, vocab, tokenizer, max_length=max_len)

    pad_idx = vocab["<pad>"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: paired_collate_fn(batch, pad_idx=pad_idx)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: paired_collate_fn(batch, pad_idx=pad_idx)
    )

    return train_loader, val_loader, vocab, tokenizer, max_len


def paired_collate_fn(batch, pad_idx):
    text1_batch, text2_batch, label_batch, fidxs = zip(*batch)
    text1_padded = pad_sequence(text1_batch, batch_first=True, padding_value=pad_idx)
    text2_padded = pad_sequence(text2_batch, batch_first=True, padding_value=pad_idx)
    labels = torch.tensor(label_batch)
    return text1_padded, text2_padded, labels, fidxs

def pad_to_min_length(padded_batch, pad_idx, min_length):
    # padded_batch: [B, T]
    seq_len = padded_batch.size(1)
    if seq_len < min_length:
        pad_amount = min_length - seq_len
        pad_tensor = torch.full((padded_batch.size(0), pad_amount), pad_idx, dtype=padded_batch.dtype)
        padded_batch = torch.cat([padded_batch, pad_tensor], dim=1)
    return padded_batch
    
def compute_length_percentiles(paired_data, tokenizer, percentile=95):
    lengths1 = []
    lengths2 = []

    for text1, text2, _ in paired_data:
        lengths1.append(len(tokenizer(text1)))
        lengths2.append(len(tokenizer(text2)))

    q4_text1 = int(np.percentile(lengths1, percentile))
    q4_text2 = int(np.percentile(lengths2, percentile))

    return q4_text1, q4_text2


def read_index_file(index_path):
    """
    Reads the index JSON file for a given bug ID and commit SHA.
    
    Returns:
        full_index (dict)
        saved_files (dict)
        path_to_file (dict)
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["full_index"], data["saved_files"], data["path_to_file"]

