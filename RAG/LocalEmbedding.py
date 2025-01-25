import numpy as np
from transformers import AutoModel, AutoTokenizer
import os
import torch
from torch import nn
from typing import List
from sentence_transformers import SentenceTransformer


class localEmbedding_sentence_dpr(nn.Module):
    def __init__(self, path: str = '', device: str = 'cuda', language_code: str = 'en_XX'):
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(path, device=device)
        self.model[0].auto_model.set_default_language(language_code)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings


class localEmbedding_sentence_ance(nn.Module):
    def __init__(self, path: str = '', device: str = 'cuda'):
        super().__init__()
        self.model = SentenceTransformer('/DR/ance', device=device)
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings


class localEmbedding_sentence_contriever(nn.Module):
    def __init__(self, path: str = '', device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy().tolist()
        
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings


class localEmbedding(nn.Module):
    def __init__(self, path: str = '', device: str = ''):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(path, add_pooling_layer=False, output_hidden_states=False)
        self.embedding.to(device)
        self.pool_type = 'cls'
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.device = device
    
    def pooling(self, token_embeddings, input):
        attention_mask = input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        input_ids = self.tokenizer(texts, max_length=256, padding="max_length", truncation=True, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        embeddings = self.embedding(**input_ids)
        if self.pool_type == 'mean':
            token_embeddings = embeddings[0]
            embeddings = self.pooling(token_embeddings, input_ids)
        else:
            embeddings = embeddings[0][:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()