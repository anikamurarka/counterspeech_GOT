import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class SentenceSimilarityScore():
    def __init__(self, use_gpu=torch.cuda.is_available()):
        self.use_gpu = use_gpu
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def similariry_score(self, text1, text2):
        # Generate embeddings for both input texts
        embedding_1 = self.model.encode(text1, convert_to_tensor=True)
        embedding_2 = self.model.encode(text2, convert_to_tensor=True)
        # Calculate cosine similarity between embeddings
        similarity = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return similarity

    def score(self, predictions, references):
        similarity_scores = []
        for idx, pred in predictions.items():
            ref = references[idx].strip()
            similarity = self.similariry_score(ref, pred)
            similarity_scores.append(similarity)
        return similarity_scores


class Bleurt():
    def __init__(self, model_path, max_length, batch_size, use_gpu=torch.cuda.is_available()):
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()

    def score(self, predictions, references):
        device = self.device
        pred_texts = []
        ref_texts = []
        for idx, pred in predictions.items():
            pred_texts.append(pred)
            ref_texts.append(references[idx])

        all_scores = []
        for i in tqdm(range(0, len(pred_texts), self.batch_size)):
            with torch.no_grad():
                batch_refs = ref_texts[i:i + self.batch_size]
                batch_preds = pred_texts[i:i + self.batch_size]
                
                if len(batch_refs) == 1:
                    continue

                inputs = self.tokenizer(batch_refs, batch_preds,
                                        return_tensors='pt', truncation=True, 
                                        padding=True, max_length=self.max_length)

                if self.use_gpu:
                    batch_scores = list(self.model(input_ids=inputs['input_ids'].to(device),
                                             attention_mask=inputs['attention_mask'].to(device),
                                             token_type_ids=inputs['token_type_ids'].to(device))[0].
                                  squeeze().cpu().numpy())
                else:
                    batch_scores = list(self.model(input_ids=inputs['input_ids'],
                                             attention_mask=inputs['attention_mask'],
                                             token_type_ids=inputs['token_type_ids'])[0].squeeze().cpu().numpy())

                all_scores += batch_scores

        return all_scores


class CounterspeechScore():
    def __init__(self, model_path, max_length, batch_size, use_gpu=torch.cuda.is_available()):
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()

    def scoring(self, texts):
        device = self.device
        text_list = [content for idx, content in texts.items()]

        result_scores = []
        for i in tqdm(range(0, len(text_list), self.batch_size)):
            with torch.no_grad():
                batch_texts = text_list[i:i + self.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', 
                                        truncation=True, padding=True,
                                        max_length=self.max_length)

                if self.use_gpu:
                    logits = self.model(input_ids=inputs['input_ids'].to(device),
                                        attention_mask=inputs['attention_mask'].to(device))[0].squeeze()
                else:
                    logits = self.model(input_ids=inputs['input_ids'], 
                                        attention_mask=inputs['attention_mask'])[0].squeeze()
                
                probabilities = torch.softmax(logits.T, dim=0).T.cpu().numpy()
                result_scores += list(probabilities[:, 1])

        return result_scores