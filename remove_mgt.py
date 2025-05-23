import argparse
import os
import pickle
import torch
from models import DualTextCNN
from data import build_paired_data_wmgt, PairedTextDataset, load_glove, simple_tokenizer, paired_collate_fn
from torch.utils.data import DataLoader

def get_hyp_mgt_list(reponame, foldername):
    hyp_list = []
    mgts_list = []
    for threshold in range(1,26):
        pos_word2count = {} 
        files = os.listdir(f"artifacts/lime_results/{reponame}_{foldername}_buggy")
        files = [x for x in files if x.endswith(".txt")]
        for file in files:
            with open(f"artifacts/lime_results/{reponame}_{foldername}_buggy/{file}",'r') as f:
                words = f.read().split('\n')[:5]
                words = [x.split('\t') for x in words if x!='']
                words = [x[0] for x in words]
            for word in words:
                if word in pos_word2count.keys():
                    pos_word2count[word]+=1
                else:
                    pos_word2count[word]=1
        pos_strong_words = list(pos_word2count.keys())
        
        word2count = {}
        files = os.listdir(f"artifacts/lime_results/{reponame}_{foldername}_nonbuggy")
        files = [x for x in files if x.endswith(".txt")]
        for file in files:
            with open(f'artifacts/lime_results/{reponame}_{foldername}_nonbuggy/{file}','r') as f:
                words = f.read().split('\n')
                words = [x.split('\t') for x in words if x!='']
                words = [x[0] for x in words]
                words = [x for x in words if x not in pos_strong_words][:threshold]
            for word in words:
                if word in word2count.keys():
                    word2count[word]+=1
                else:
                    word2count[word]=1
        if len(word2count)==0:
            continue
        line = min(100, len(word2count))
        for t2 in range(1,line):
            hyp_list.append((threshold, t2))
            mgt = list(word2count.keys())[:t2]
            mgts_list.append(mgt)
    return hyp_list, mgts_list

def run_all_hyp_list(reponame, model_dir, hyp_list, mgts_list, max_length):

    # === Load Model ===
    model = DualTextCNN(embed_dim=100, num_classes=2, kernel_sizes=(3,4,5), num_channels=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))

    # === Load Dataset ===
    test_indexing_dir = f"datasets/SWEBench_Custom/{reponame}_test_indexing"
    test_bugs_dir = f"datasets/SWEBench_Custom/{reponame}_test_bugs/{reponame}_bugs.pkl"
    with open(test_bugs_dir ,'rb') as f:
        test_bugs = pickle.load(f)

    # === Run model and rank files by predicted P(label=1) ===
    model.eval()
    with open(f"{reponame}_stop_nltk_word_mgt_result.txt",'w') as f:
        pass
    for hy, mgts in zip(hyp_list, mgts_list):
        print("Testing", hy, len(mgts) )
        global_hits_at_1 = 0
        global_hits_at_5 = 0
        global_mrr_total = 0
        global_map_total = 0
        bug_count_with_positives = 0
        
        for bug in test_bugs:
            brid = bug['number']
            #paired_data, _ = build_paired_data([bug], test_indexing_dir)
            #build_paired_data_wmgt(mgt)
            paired_data, _, fidxs = build_paired_data_wmgt([bug], test_indexing_dir, mgts)
        
            # === Setup Dataset and Loader ===
            vocab, _ = load_glove(embedding_dim=100)
            dataset = PairedTextDataset(
                data=paired_data,
                vocab=vocab,
                tokenizer=simple_tokenizer,
                max_length=max_length,
                fidxs=fidxs
            )
        
            test_loader = DataLoader(
                dataset,
                batch_size=1024,
                shuffle=False,
                collate_fn=lambda batch: paired_collate_fn(batch, pad_idx=vocab["<pad>"])
            )
            
            ### Calculate the rank of data with label 1 ###
            scores = []
            labels = []
        
            with torch.no_grad():
                for x1, x2, y,_ in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    logits = model(x1, x2)
                    probs = torch.softmax(logits, dim=1)[:, 1]  # P(label=1)
                    scores.extend(probs.cpu().tolist())
                    labels.extend(y.cpu().tolist())
        
            # === Rank files by score ===
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            true_indices = {i for i, label in enumerate(labels) if label == 1}
            num_true = len(true_indices)
            
            if num_true == 0:
                continue  # skip bug with no fixed files
            
            bug_count_with_positives += 1
        
            positive_ranks = []
            for rank, (idx, _) in enumerate(ranked, 1):
                if labels[idx] == 1:
                    positive_ranks.append(rank)
        
            if positive_ranks:
                avg_rank = sum(positive_ranks) / len(positive_ranks)
                #print(f"[{brid}] Ranks of Fixed Files: {positive_ranks}")
            else:
                print(f"[{brid}] No fixed files found in this bug report.")
        
        
            # === Precision@K, MRR, MAP ===
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
            ###########
        
        if bug_count_with_positives > 0:
            avg_p1 = global_hits_at_1 / bug_count_with_positives
            avg_p5 = global_hits_at_5 / bug_count_with_positives # Precision: (bug_count_with_positives * 5)
            avg_mrr = global_mrr_total / bug_count_with_positives
            avg_map = global_map_total / bug_count_with_positives
        
            print(f"\n=== Global Evaluation Across {bug_count_with_positives} Bugs, {hy}, {len(mgts)} ===")
            print(f"Acc@1: {avg_p1:.4f}")
            print(f"Acc@5: {avg_p5:.4f}")
            print(f"MAP:         {avg_map:.4f}")
            print(f"MRR:         {avg_mrr:.4f}")

            with open(f"{reponame}_stop_nltk_word_mgt_result.txt",'a') as f:
                f.write(f"\n=== Global Evaluation Across {bug_count_with_positives} Bugs, {hy}, {len(mgts)} ===\n")
                f.write(f"Acc@1: {avg_p1:.4f}\n")
                f.write(f"Acc@5: {avg_p5:.4f}\n")
                f.write(f"MAP:         {avg_map:.4f}\n")
                f.write(f"MRR:         {avg_mrr:.4f}\n")
        else:
            print("No bugs with label 1 found.")


#threshold, t2 = 6, 96
def run_with_designated_threshold(model_dir, threshold, t2, test_bugs):

    # Load model
    model = DualTextCNN(embed_dim=100, num_classes=2, kernel_sizes=(3,4,5), num_channels=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    
    # Collect MGTs
    pos_word2count = {}
    files = os.listdir("stop_nltk_lime_buggy")
    files = [x for x in files if x.endswith(".txt")]
    for file in files:
        with open(f'stop_nltk_lime_buggy/{file}','r') as f:
            words = f.read().split('\n')[:5]
            words = [x.split('\t') for x in words if x!='']
            words = [x[0] for x in words]
        for word in words:
            if word in pos_word2count.keys():
                pos_word2count[word]+=1
            else:
                pos_word2count[word]=1
    pos_strong_words = list(pos_word2count.keys())

    word2count = {}
    files = os.listdir("stop_nltk_lime_nonbuggy")
    files = [x for x in files if x.endswith(".txt")]
    for file in files:
        with open(f'stop_nltk_lime_nonbuggy/{file}','r') as f:
            words = f.read().split('\n')
            words = [x.split('\t') for x in words if x!='']
            words = [x[0] for x in words]
            words = [x for x in words if x not in pos_strong_words][:threshold]
        for word in words:
            if word in word2count.keys():
                word2count[word]+=1
            else:
                word2count[word]=1


    mgts = list(word2count.keys())[:t2]
    print(len(mgts))


    # Apply MGTs
    global_hits_at_1 = 0
    global_hits_at_5 = 0
    global_mrr_total = 0
    global_map_total = 0
    bug_count_with_positives = 0

    brid2rankedsf = {}
    for bug in test_bugs:
        brid = bug['number']
        paired_data, _, fidxs = build_paired_data_wmgt([bug], test_indexing_dir, mgts)

        # === Setup Dataset and Loader ===
        dataset = PairedTextDataset(
            data=paired_data,
            vocab=vocab,
            tokenizer=nltk_word_tokenizer,
            max_length=max_length,
            fidxs=fidxs
        )

        test_loader = DataLoader(
            dataset,
            batch_size=512,
            shuffle=False,
            collate_fn=lambda batch: paired_collate_fn(batch, pad_idx=vocab["<pad>"])
        )
        
        ### Calculate the rank of data with label 1 ###
        scores = []
        labels = []
        fidxs_seq = []

        with torch.no_grad():
            for x1, x2, y, fidx in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(label=1)
                scores.extend(probs.cpu().tolist())
                labels.extend(y.cpu().tolist())
                fidxs_seq.append(fidx)

        # === Rank files by score ===
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        true_indices = {i for i, label in enumerate(labels) if label == 1}
        num_true = len(true_indices)
        
        if num_true == 0:
            continue  # skip bug with no fixed files
        
        bug_count_with_positives += 1

        positive_ranks = []
        for rank, (idx, _) in enumerate(ranked, 1):
            if labels[idx] == 1:
                positive_ranks.append(rank)

        if positive_ranks:
            avg_rank = sum(positive_ranks) / len(positive_ranks)
            #print(f"[{brid}] Ranks of Fixed Files: {positive_ranks}")
        else:
            print(f"[{brid}] No fixed files found in this bug report.")

        # === Save ranked files by score ===
        ranked_fidx = []
        for rank, (idx, _) in enumerate(ranked, 1):
            fidx = fidxs[idx]
            ranked_fidx.append(fidx)
        brid2rankedsf[brid] = ranked_fidx
            

        # === Precision@K, MRR, MAP ===
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
        ###########

    with open("brid2rankedsf.pkl",'wb') as f:
        pickle.dump(brid2rankedsf, f)
        
    if bug_count_with_positives > 0:
        avg_p1 = global_hits_at_1 / bug_count_with_positives
        avg_p5 = global_hits_at_5 / bug_count_with_positives # Precision: (bug_count_with_positives * 5)
        avg_mrr = global_mrr_total / bug_count_with_positives
        avg_map = global_map_total / bug_count_with_positives

        print(f"\n=== Global Evaluation Across {bug_count_with_positives} Bugs, {len(mgts)} ===")
        print(f"Acc@1: {avg_p1:.4f}")
        print(f"Acc@5: {avg_p5:.4f}")
        print(f"MAP:         {avg_map:.4f}")
        print(f"MRR:         {avg_mrr:.4f}")


parser = argparse.ArgumentParser(description="Reponame and target model")
parser.add_argument("--foldername", type=str, default="stop_nltk_lime")
parser.add_argument("--reponame", type=str, required=True, help="Repository name (e.g., sphinx-doc+sphinx)")
parser.add_argument("--model_dir", type=str, default="artifacts/trained_model/best_model.pt", help="Path to save/load model")
parser.add_argument("--max_length", type=int, default=4102, help="Maximum sequence length")

args = parser.parse_args()


hyp_list, mgts_list = get_hyp_mgt_list(args.reponame, args.foldername)
run_all_hyp_list(args.reponame, args.model_dir, hyp_list, mgts_list, args.max_length)