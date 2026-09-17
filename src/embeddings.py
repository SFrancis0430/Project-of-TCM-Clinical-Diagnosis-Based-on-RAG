"""本地 Embedding：直接用 transformers 加载 text2vec-base-chinese 缓存，均值池化。

为什么不用 sentence-transformers / HuggingFaceEmbeddings：
本机缓存的模型是旧版 sentence-transformers 格式（sentence_bert_config.json），
新版 ST 会联网拉取 config_sentence_transformers.json，在无外网环境下会卡死。
这里改用 transformers 直接加载 BERT 权重 + 均值池化（text2vec 即 BERT+mean pooling），
既离线可用，也便于面试时解释 embedding 的底层原理。
"""
from functools import lru_cache

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

import config

class LocalText2VecEmbeddings(Embeddings):
    """text2vec-base-chinese 的本地实现：BERT + mean pooling + L2 归一化。"""

    def __init__(self, model_name: str, local_only: bool = True, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self._model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """对 token 向量按 attention_mask 加权平均，排除 padding。"""
        mask = attention_mask.unsqueeze(-1).float()
        s = (hidden * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp(min=1e-9)
        return s / d

    def _embed(self, texts: list[str]) -> np.ndarray:
        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
            vecs = self._mean_pool(hidden, inputs["attention_mask"])
        vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        return vecs.cpu().numpy()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.append(self._embed(texts[i : i + self.batch_size]))
        return np.vstack(out).tolist() if out else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> "LocalText2VecEmbeddings":
    """Embedding 单例（模块级缓存，避免重复加载权重）。"""
    return LocalText2VecEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        local_only=config.EMBEDDING_LOCAL_ONLY,
    )
