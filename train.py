import argparse
import torch
from models import DualTextCNN
from training import Trainer
from data import load_and_prepare_bug_data, prepare_dataloaders
from evaluation.ranker import evaluate_bug_ranking

def run_pipeline(
    evaluate_only,
    repodir,
    reponame,
    model_name,
    embed_dim,
    kernel_sizes,
    num_channels,
    patience,
    max_length,
    lr
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not evaluate_only: # Train
        tfidxs, vfidxs, train_paired_data, train_weights, val_paired_data, val_weights = load_and_prepare_bug_data(reponame, repodir)
        train_loader, val_loader, vocab, tokenizer, max_length = prepare_dataloaders(
            tfidxs,
            vfidxs,
            train_paired_data,
            val_paired_data,
            batch_size=32,
            embedding_dim=embed_dim
        )

        model = DualTextCNN(embed_dim=embed_dim, kernel_sizes=kernel_sizes, num_channels=num_channels)
        trainer = Trainer(model, train_weights, lr)
        trainer.set_validation_weights(val_weights)

        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(100):
            train_loss = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.evaluate(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                trainer.save_model(model_name)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    # Evaluation
    model = DualTextCNN(embed_dim=embed_dim, kernel_sizes=kernel_sizes, num_channels=num_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))


    #model.load_state_dict(torch.load(model_name, map_location=device, weights_only=True))
    evaluate_bug_ranking(model, reponame, repodir, max_length)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DualTextCNN for bug localization")

    parser.add_argument("--evaluate_only", action="store_true", help="Run evaluation only, skip training")
    parser.add_argument("--repodir", type=str, required=True, help="Path to the base repo directory")
    parser.add_argument("--reponame", type=str, required=True, help="Repository name (e.g., sphinx-doc+sphinx)")
    parser.add_argument("--model_name", type=str, default="artifacts/trained_model/best_model.pt", help="Path to save/load model")
    parser.add_argument("--embed_dim", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[3, 4, 5], help="Convolution kernel sizes")
    parser.add_argument("--num_channels", type=int, default=100, help="Number of channels per convolution kernel")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=4102, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pipeline(
        evaluate_only=args.evaluate_only,
        repodir=args.repodir,
        reponame=args.reponame,
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        kernel_sizes=tuple(args.kernel_sizes),
        num_channels=args.num_channels,
        patience=args.patience,
        max_length=args.max_length,
        lr=args.lr
    )
