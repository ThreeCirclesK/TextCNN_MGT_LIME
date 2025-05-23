import os
import pickle
import torch
from torch.utils.data import DataLoader
from data import (
    build_paired_data,
    PairedTextDataset,
    simple_tokenizer,
    load_glove,
    paired_collate_fn,
)

def evaluate_bug_ranking(model, reponame, repodir, max_length=4102, glove_path='artifacts/glove6B', embedding_dim = 100, device=None):
    model.eval()
    
    repo_path = f"{repodir}/{reponame}"
    test_indexing_dir = os.path.abspath(f"{repo_path}_test_indexing")
    
    with open(f"artifacts/test_bugs/{reponame}_bugs.pkl", 'rb') as f:
        test_bugs = pickle.load(f)

    test_paired_data, _, fidxs = build_paired_data(test_bugs, test_indexing_dir, brid2commit=None)  # brid2commit unused for test bugs
    vocab, _ = load_glove(embedding_dim)

    global_hits_at_1 = 0
    global_hits_at_5 = 0
    global_mrr_total = 0
    global_map_total = 0
    bug_count_with_positives = 0

    for bug in test_bugs:
        brid = bug['number']
        paired_data, _, fidxs= build_paired_data([bug], test_indexing_dir, brid2commit=None)

        if not paired_data:
            continue

        dataset = PairedTextDataset(
            fidxs = fidxs,
            data=paired_data,
            vocab=vocab,
            tokenizer=simple_tokenizer,
            max_length=max_length
        )

        test_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: paired_collate_fn(batch, pad_idx=vocab["<pad>"])
        )

        scores = []
        labels = []
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for x1, x2, y, _ in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                probs = torch.softmax(logits, dim=1)[:, 1]
                scores.extend(probs.cpu().tolist())
                labels.extend(y.cpu().tolist())

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        true_indices = {i for i, label in enumerate(labels) if label == 1}
        num_true = len(true_indices)

        if num_true == 0:
            continue

        bug_count_with_positives += 1
        positive_ranks = []
        for rank, (idx, _) in enumerate(ranked, 1):
            if labels[idx] == 1:
                positive_ranks.append(rank)


        print(f"[{brid}] Ranks of Fixed Files: {positive_ranks}" if positive_ranks else f"[{brid}] No fixed files found.")

        # Acc@1
        if ranked[0][0] in true_indices:
            global_hits_at_1 += 1

        # Acc@5
        top5 = [idx for idx, _ in ranked[:5]]
        global_hits_at_5 += sum(1 for i in top5 if i in true_indices)

        # MRR
        for rank, (idx, _) in enumerate(ranked, 1):
            if idx in true_indices:
                global_mrr_total += 1.0 / rank
                break

        # MAP
        ap_sum = 0.0
        hits = 0
        for rank, (idx, _) in enumerate(ranked, 1):
            if idx in true_indices:
                hits += 1
                ap_sum += hits / rank
        global_map_total += ap_sum / num_true

    if bug_count_with_positives > 0:
        avg_p1 = global_hits_at_1 / bug_count_with_positives
        avg_p5 = global_hits_at_5 / bug_count_with_positives
        avg_mrr = global_mrr_total / bug_count_with_positives
        avg_map = global_map_total / bug_count_with_positives

        print(f"\n=== Global Evaluation Across {bug_count_with_positives} Bugs ===")
        print(f"Acc@1: {avg_p1:.4f}")
        print(f"Acc@5: {avg_p5:.4f}")
        print(f"MAP:   {avg_map:.4f}")
        print(f"MRR:   {avg_mrr:.4f}")
    else:
        print("No bugs with label 1 found.")
