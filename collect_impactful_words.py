import argparse
import torch
from lime.lime_text import LimeTextExplainer
from models import DualTextCNN, LimeDualWrapper
from training import Trainer
from data import load_and_prepare_bug_data, prepare_dataloaders, read_index_file, build_paired_data
from evaluation.ranker import evaluate_bug_ranking
from tqdm.auto import tqdm
import os
import pickle
from data import (PairedTextDataset, load_glove, simple_tokenizer, paired_collate_fn, pad_to_min_length)
from torch.utils.data import Dataset, DataLoader

def collect_lime(repo_path, reponame, model_dir, max_length):
    with open(f"datasets/past_brid2commit/{reponame}_bugs_brid2commit.pkl",'rb') as f:
        bugs, brid2commit = pickle.load(f)
        print(f"{len(bugs)} past BRs, {len(brid2commit)} BRs with mapped commits")
        bugs = [x for x in bugs if x['number'] in brid2commit.keys()]
        indexing_dir = os.path.abspath(f"{repo_path}/{reponame}_indexing")


    model = DualTextCNN(embed_dim=100, num_classes=2, kernel_sizes=(3,4,5), num_channels=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.eval()

    for bug in tqdm(bugs, desc="Processing bugs"):
        brid = bug['number']
        if 'title' in bug.keys(): ## Train & Validation Dataset
            title = bug['title']
            body = bug['body']
            brtext = f"{title}\n{body}"
            commit_sha = brid2commit[brid]
        else: ## Test Dataset
            brtext = bug['brtext']
            commit_sha = bug['parent_commit'] 

        index_filename = f"{brid}_{commit_sha}.json"
        index_path = os.path.join(indexing_dir, index_filename)
        idx2file, _, file2idx = read_index_file(index_path)
            
        paired_data, _, fidxs = build_paired_data([bug], indexing_dir, brid2commit)

        # === Setup Dataset ===
        vocab, _ = load_glove(embedding_dim=100)
        dataset = PairedTextDataset(
            fidxs = fidxs,
            data=paired_data,
            vocab=vocab,
            tokenizer=simple_tokenizer,
            max_length=max_length
        )

        test_loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda batch: paired_collate_fn(batch, pad_idx=vocab["<pad>"])
        )
        
        ### Calculate the rank of data with label 1 ###
        scores = []
        labels = []
        fidxs = []

        with torch.no_grad():
            for x1, x2, y, fidx in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(label=1)
                scores.extend(probs.cpu().tolist())
                labels.extend(y.cpu().tolist())
                fidxs+=list(fidx)

        # === Rank files by score ===
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        true_indices = {i for i, label in enumerate(labels) if label == 1}
        num_true = len(true_indices)
        
        if num_true == 0:
            continue  # skip bug with no fixed files

        # === Pick misjudged wrong buggy files for LIME explanation ===
        best_rank, best_idx, best_score = None, None, None
        for rank, (idx, score) in enumerate(ranked, 1):
            if labels[idx] == 1:
                best_rank, best_idx, best_score = rank, idx, score
                break

        target_fidxs = []
        buggy_fidxs = []
        for s,l,f in zip(scores, labels, fidxs):
            if l==1:
                buggy_fidxs.append(f)
            else:
                if s>best_score:
                    assert type(f)==str
                    target_fidxs.append(f)

        if len(target_fidxs)==0:
            continue
            
        print(len(target_fidxs))
        # === Pick first fixed file for LIME explanation ===
        for fidx in target_fidxs:
            file_path = os.path.join(indexing_dir, fidx)
            with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
                sftext = f.read()

            explainer = LimeTextExplainer(class_names=["Not Fixed", "Fixed"])

            # Analyze BR text
            wrapper = LimeDualWrapper(model, vocab, simple_tokenizer, fixed_text=sftext, 
                                    device=device, analyze='input1',min_length=5, batch_size=128)
            explanation = explainer.explain_instance(brtext, wrapper.predict, num_features=25, num_samples=1000, labels=[0, 1])
            explanation.save_to_file(f"artifacts/lime_results/{reponame}_stop_nltk_lime_nonbuggy/lime_explanation_{brid}_{fidx}_br.html")
            with open(f"artifacts/lime_results/{reponame}_stop_nltk_lime_nonbuggy/lime_explanation_{brid}_{fidx}_br.txt",'w',encoding='utf8',errors='ignore') as f:
                words = explanation.as_list(label=1)
                for word, score in words:
                    f.write(f'{word}\t{score}\n')

            # Analyze SF text
            wrapper = LimeDualWrapper(model, vocab, simple_tokenizer, fixed_text=brtext, 
                                    device=device, analyze='input2', min_length=5, batch_size=128)
            explanation = explainer.explain_instance(sftext, wrapper.predict, num_features=50, num_samples=1000, labels=[0, 1])
            explanation.save_to_file(f"artifacts/lime_results/{reponame}_stop_nltk_lime_nonbuggy/lime_explanation_{brid}_{fidx}_sf.html")
            with open(f"artifacts/lime_results/{reponame}_stop_nltk_lime_nonbuggy/lime_explanation_{brid}_{fidx}_sf.txt",'w',encoding='utf8',errors='ignore') as f:
                words = explanation.as_list(label=1)
                for word, score in words:
                    f.write(f'{word}\t{score}\n')
        
            print(f"[{brid}][{fidx}] LIME explanation saved.")
            torch.cuda.empty_cache()

        for fidx in buggy_fidxs:
            file_path = os.path.join(indexing_dir, fidx)
            with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
                sftext = f.read()

            explainer = LimeTextExplainer(class_names=["Not Fixed", "Fixed"])

            # Analyze BR text
            wrapper = LimeDualWrapper(model, vocab, simple_tokenizer, fixed_text=sftext, 
                                    device=device, analyze='input1',min_length=5, batch_size=128)
            explanation = explainer.explain_instance(brtext, wrapper.predict, num_features=25, num_samples=1000, labels=[0, 1])
            explanation.save_to_file(f"artifacts/lime_results/{reponame}_stop_nltk_lime_buggy/lime_explanation_{brid}_{fidx}_br.html")
            with open(f"artifacts/lime_results/{reponame}_stop_nltk_lime_buggy/lime_explanation_{brid}_{fidx}_br.txt",'w',encoding='utf8',errors='ignore') as f:
                words = explanation.as_list(label=1)
                for word, score in words:
                    f.write(f'{word}\t{score}\n')

            # Analyze SF text
            wrapper = LimeDualWrapper(model, vocab, simple_tokenizer, fixed_text=brtext, 
                                    device=device, analyze='input2', min_length=5, batch_size=128)
            explanation = explainer.explain_instance(sftext, wrapper.predict, num_features=50, num_samples=1000, labels=[0, 1])
            explanation.save_to_file(f"artifacts/lime_results/{reponame}_stop_nltk_lime_buggy/lime_explanation_{brid}_{fidx}_sf.html")
            with open(f"artifacts/lime_results/{reponame}_stop_nltk_lime_buggy/lime_explanation_{brid}_{fidx}_sf.txt",'w',encoding='utf8',errors='ignore') as f:
                words = explanation.as_list(label=1)
                for word, score in words:
                    f.write(f'{word}\t{score}\n')
        
            print(f"[{brid}][{fidx}] LIME explanation saved.")
            torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description="Reponame and target model")
parser.add_argument("--repodir", type=str, required=True, help="Path to the base repo directory")
parser.add_argument("--reponame", type=str, required=True, help="Repository name (e.g., sphinx-doc+sphinx)")
parser.add_argument("--model_dir", type=str, default="artifacts/trained_model/best_model.pt", help="Path to save/load model")
parser.add_argument("--max_length", type=int, default=4102, help="Maximum sequence length")

args = parser.parse_args()

# Directory to save lime results
if not os.path.isdir(f"artifacts/lime_results/{args.reponame}_stop_nltk_lime_nonbuggy"):
    os.makedirs(f"artifacts/lime_results/{args.reponame}_stop_nltk_lime_nonbuggy")
if not os.path.isdir(f"artifacts/lime_results/{args.reponame}_stop_nltk_lime_buggy"):
    os.makedirs(f"artifacts/lime_results/{args.reponame}_stop_nltk_lime_buggy")


collect_lime(args.repodir, args.reponame, args.model_dir, args.max_length)