"""
Four Siamese (shared-weight) document encoders for resume-JD matching.
Each encoder maps a token-id sequence (or raw text, for SBERT) to a
fixed-length vector. Fit score = cosine similarity between the resume
vector and the JD vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseMLP(nn.Module):
    """Bag-of-averaged-embeddings MLP baseline. No sequence information."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout_in = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout_out = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pad_id = pad_id

    def encode(self, x):
        mask = (x != self.pad_id).unsqueeze(-1).float()
        emb = self.embedding(x)
        emb = self.dropout_in(emb)
        summed = (emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        avg = summed / count
        h = F.relu(self.fc1(avg))
        h = self.dropout_out(h)
        h = self.fc2(h)
        return h

    def forward(self, resume_ids, jd_ids):
        r = self.encode(resume_ids)
        j = self.encode(jd_ids)
        return F.cosine_similarity(r, j, dim=-1)


class SiameseCNN(nn.Module):
    """Multi-kernel (3/4/5) 1D-CNN encoder, Kim (2014)-style."""

    def __init__(self, vocab_size, embed_dim=128, num_filters=64, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout_in = nn.Dropout(0.1)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding="same")
            for k in (3, 5, 7)
        ])
        for conv in self.convs:
            # Default PyTorch Conv1d init is too large for this embed_dim
            # (128) -> num_filters (64) fan-in and tends to land the model
            # in a degenerate "constant output" basin on this task; a
            # smaller init gives gradients enough room to actually
            # differentiate between inputs early in training.
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            conv.weight.data.mul_(0.3)
            nn.init.zeros_(conv.bias)
        self.dropout_mid = nn.Dropout(0.1)
        self.fc1 = nn.Linear(num_filters * 3, 64)
        self.out_norm = nn.LayerNorm(64)
        self.dropout_out = nn.Dropout(0.1)
        self.pad_id = pad_id

    def encode(self, x):
        pad_mask = (x == self.pad_id).unsqueeze(1)  # (batch, 1, seq_len), True where padded
        emb = self.embedding(x)
        emb = self.dropout_in(emb)
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))  # (batch, filters, seq_len)
            c = c.masked_fill(pad_mask, float("-inf"))
            p = c.max(dim=2).values
            pooled.append(p)
        h = torch.cat(pooled, dim=-1)
        h = self.dropout_mid(h)
        h = F.relu(self.fc1(h))
        h = self.out_norm(h)
        h = self.dropout_out(h)
        return h

    def forward(self, resume_ids, jd_ids):
        r = self.encode(resume_ids)
        j = self.encode(jd_ids)
        return F.cosine_similarity(r, j, dim=-1)


class SiameseBiLSTM(nn.Module):
    """Bidirectional LSTM encoder with long-range sequence memory."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout_in = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True,
            bidirectional=True, dropout=0.0,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.out_norm = nn.LayerNorm(64)
        self.dropout_out = nn.Dropout(0.4)
        self.pad_id = pad_id

    def encode(self, x):
        mask = (x != self.pad_id)
        lengths = mask.sum(dim=1).clamp(min=1)
        emb = self.embedding(x)
        emb = self.dropout_in(emb)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )  # (batch, seq_len, hidden_dim*2)
        mask_f = mask.unsqueeze(-1).float()
        summed = (out * mask_f).sum(dim=1)
        h = summed / lengths.unsqueeze(-1).float()  # mean-pool over valid (non-pad) timesteps
        h = F.relu(self.fc1(h))
        h = self.out_norm(h)
        h = self.dropout_out(h)
        return h

    def forward(self, resume_ids, jd_ids):
        r = self.encode(resume_ids)
        j = self.encode(jd_ids)
        return F.cosine_similarity(r, j, dim=-1)


class SiameseSBERT(nn.Module):
    """
    Fine-tuned pretrained sentence-transformer bi-encoder. Wraps a small
    pretrained checkpoint (all-MiniLM-L6-v2) rather than full DistilBERT,
    to keep fine-tuning fast on this hardware/time budget; still a real
    pretrained Transformer fine-tuned end-to-end with the same
    cosine-similarity objective as the other three models.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def encode_texts(self, texts, device):
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        ).to(device)
        out = self.backbone(**enc)
        return self._mean_pool(out, enc["attention_mask"])

    def forward(self, resume_texts, jd_texts, device):
        r = self.encode_texts(resume_texts, device)
        j = self.encode_texts(jd_texts, device)
        return F.cosine_similarity(r, j, dim=-1)
