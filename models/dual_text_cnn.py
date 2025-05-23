import torch
import torch.nn as nn
from data.glove_loader import load_glove
    
class DualTextCNN(nn.Module):
    def __init__(self, embed_dim=100, num_classes=2, kernel_sizes=(3, 4, 5), num_channels=100):
        super().__init__()
        vocab, embedding_matrix =  load_glove(embedding_dim=embed_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=vocab["<pad>"])
        #self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(embed_dim, num_channels, k) for k in kernel_sizes
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(embed_dim, num_channels, k) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels * len(kernel_sizes) * 2, num_classes)

    def encode(self, x, convs):
        x = self.embedding(x).permute(0, 2, 1)  # [B, D, T]
        x = [torch.relu(conv(x)) for conv in convs]  # List of [B, C, T']
        x = [torch.max(c, dim=2)[0] for c in x]  # List of [B, C]
        return torch.cat(x, dim=1)  # [B, C * len(kernel_sizes)]

    def forward(self, input1, input2):
        out1 = self.encode(input1, self.convs1)
        out2 = self.encode(input2, self.convs2)
        combined = torch.cat([out1, out2], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)

