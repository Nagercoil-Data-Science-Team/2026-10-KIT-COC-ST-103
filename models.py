"""
Neural Architecture Suite for Automatic Essay Scoring (AES)
Implements:
1. Spike-Gated Self-Attention (SGSA) Module
2. SpikeBERT-SHHO (Proposed Model)
3. Attention-based Recurrent Convolutional Neural Network (Base Paper Model)
4. Baselines: CNN, RNN/LSTM, CNN-LSTM, Standard BERT, BERT+SGSA
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Locate cached local BERT model or fall back to HuggingFace hub id
LOCAL_BERT_SNAPSHOT = r"C:\Users\ADMIN\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
if os.path.exists(LOCAL_BERT_SNAPSHOT):
    LOCAL_BERT_PATH = LOCAL_BERT_SNAPSHOT
else:
    LOCAL_BERT_PATH = "sentence-transformers/all-MiniLM-L6-v2"


class SpikeGatedSelfAttention(nn.Module):
    """
    Step 4: Spike-Gated Self-Attention (SGSA)
    Combines multi-head contextual self-attention with bio-inspired
    spike threshold gating to selectively activate salient semantic tokens.
    """
    def __init__(self, embed_dim, num_heads=4, spike_threshold=0.35, tau=0.1):
        super(SpikeGatedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable/optimizable spike gating parameters
        self.spike_threshold = nn.Parameter(torch.tensor(float(spike_threshold)))
        self.tau = tau
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        mask: [batch_size, seq_len] (1 for valid, 0 for pad)
        """
        B, L, D = x.shape
        residual = x
        
        # Linear projections & split into multi-heads
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, L, d]
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, H, L, L]
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1) # [B, H, L, L]
        u = torch.matmul(attn_weights, v) # [B, H, L, d]
        u = u.transpose(1, 2).contiguous().view(B, L, D) # [B, L, D]
        
        # Spike gating activation:
        # Computes surrogate gradient spike excitation: S = sigmoid((u - theta) / tau)
        membrane_potential = u - self.spike_threshold
        spike_gate = torch.sigmoid(membrane_potential / self.tau)
        
        # Selectively gated representation with residual bypass
        gated_features = spike_gate * u
        out = self.norm(residual + self.out_proj(gated_features))
        return out, attn_weights


class SpikeBERTModel(nn.Module):
    """
    Proposed SpikeBERT Architecture:
    BERT contextual representation -> Spike-Gated Self-Attention (SGSA) ->
    Attentive Essay Pooling -> Score Regression Head.
    """
    def __init__(self, bert_path=LOCAL_BERT_PATH, spike_threshold=0.35, dropout=0.25, hidden_dim=128):
        super(SpikeBERTModel, self).__init__()
        local_only = os.path.exists(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path, local_files_only=local_only)
        # Freeze lower layers to accelerate CPU fine-tuning while fine-tuning top layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:-2]:
            for param in layer.parameters():
                param.requires_grad = False
                
        embed_dim = self.bert.config.hidden_size
        self.sgsa = SpikeGatedSelfAttention(embed_dim, num_heads=4, spike_threshold=spike_threshold)
        
        # Attentive pooling layer
        self.pool_proj = nn.Linear(embed_dim, 1)
        
        # Score regression head
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output normalized score between 0.0 and 1.0
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_features = bert_out.last_hidden_state # [B, L, D]
        
        # Spike-Gated Self-Attention
        gated_features, _ = self.sgsa(token_features, mask=attention_mask)
        
        # Attentive feature aggregation
        scores = self.pool_proj(gated_features).squeeze(-1) # [B, L]
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        alpha = F.softmax(scores, dim=-1).unsqueeze(-1) # [B, L, 1]
        essay_rep = torch.sum(gated_features * alpha, dim=1) # [B, D]
        
        # Score prediction
        pred = self.regressor(essay_rep).squeeze(-1) # [B]
        return pred


class StandardBERTModel(nn.Module):
    """BERT without SGSA (Ablation / Baseline)"""
    def __init__(self, bert_path=LOCAL_BERT_PATH, dropout=0.25, hidden_dim=128):
        super(StandardBERTModel, self).__init__()
        local_only = os.path.exists(bert_path)
        self.bert = AutoModel.from_pretrained(bert_path, local_files_only=local_only)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:-2]:
            for param in layer.parameters():
                param.requires_grad = False
                
        embed_dim = self.bert.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_features = bert_out.last_hidden_state
        # Mean pooling with mask
        mask_expanded = attention_mask.unsqueeze(-1).float()
        essay_rep = (token_features * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pred = self.regressor(essay_rep).squeeze(-1)
        return pred


class AttentionRCNN(nn.Module):
    """
    Base Paper Model:
    Attention-Based Recurrent Convolutional Neural Network for Automatic Essay Scoring
    (Word Embedding -> CNN -> BiLSTM -> Attention Mechanism -> Score Regression)
    """
    def __init__(self, vocab_size=30522, embed_dim=128, cnn_filters=128, lstm_hidden=64, dropout=0.3):
        super(AttentionRCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(lstm_hidden * 2, 1)
        
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids) # [B, L, D]
        conv_out = F.relu(self.conv1(emb.transpose(1, 2))).transpose(1, 2) # [B, L, C]
        lstm_out, _ = self.bilstm(conv_out) # [B, L, 2H]
        
        attn_scores = self.attn_proj(lstm_out) # [B, L, 1]
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1) # [B, L, 1]
        essay_rep = torch.sum(lstm_out * attn_weights, dim=1) # [B, 2H]
        pred = self.regressor(essay_rep).squeeze(-1)
        return pred


class CNNModel(nn.Module):
    """CNN Baseline for Essay Scoring"""
    def __init__(self, vocab_size=30522, embed_dim=128, cnn_filters=128, dropout=0.3):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2)
        self.regressor = nn.Sequential(
            nn.Linear(cnn_filters, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids).transpose(1, 2)
        x = F.relu(self.conv1(emb))
        x = F.relu(self.conv2(x))
        essay_rep = torch.max(x, dim=2)[0]
        pred = self.regressor(essay_rep).squeeze(-1)
        return pred


class RNNLSTMModel(nn.Module):
    """RNN/LSTM Baseline for Essay Scoring"""
    def __init__(self, vocab_size=30522, embed_dim=128, lstm_hidden=64, dropout=0.3):
        super(RNNLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(emb)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            essay_rep = (lstm_out * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        else:
            essay_rep = torch.mean(lstm_out, dim=1)
        pred = self.regressor(essay_rep).squeeze(-1)
        return pred


class CNNLSTMModel(nn.Module):
    """CNN-LSTM Hybrid Baseline for Essay Scoring"""
    def __init__(self, vocab_size=30522, embed_dim=128, cnn_filters=128, lstm_hidden=64, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, cnn_filters, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True, bidirectional=True)
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids).transpose(1, 2)
        conv_out = F.relu(self.conv(emb)).transpose(1, 2)
        lstm_out, _ = self.bilstm(conv_out)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            essay_rep = (lstm_out * mask_expanded).sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        else:
            essay_rep = torch.mean(lstm_out, dim=1)
        pred = self.regressor(essay_rep).squeeze(-1)
        return pred


class NonBERT_SGSA_SHHO(nn.Module):
    """Ablation Variant: Without BERT (BiLSTM + SGSA + SHHO)"""
    def __init__(self, vocab_size=30522, embed_dim=128, lstm_hidden=192, spike_threshold=0.35, dropout=0.25):
        super(NonBERT_SGSA_SHHO, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        feat_dim = lstm_hidden * 2
        self.sgsa = SpikeGatedSelfAttention(feat_dim, num_heads=4, spike_threshold=spike_threshold)
        self.pool_proj = nn.Linear(feat_dim, 1)
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(emb)
        gated_features, _ = self.sgsa(lstm_out, mask=attention_mask)
        scores = self.pool_proj(gated_features).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        alpha = F.softmax(scores, dim=-1).unsqueeze(-1)
        essay_rep = torch.sum(gated_features * alpha, dim=1)
        pred = self.regressor(essay_rep).squeeze(-1)
        return pred
