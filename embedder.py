import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "law-ai/InLegalBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

@torch.no_grad()
def embed_text(text: str):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    out = model(**tokens)
    emb = out.last_hidden_state.mean(dim=1).squeeze()
    emb = emb / emb.norm()

    # ✅ critical: convert to python list
    return emb.cpu().numpy().astype(float).tolist()


def token_len(text: str):
    return len(tokenizer.encode(text))
