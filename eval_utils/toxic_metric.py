import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertPreTrainedModel, BertModel


class CustomBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_fn = nn.Tanh()

    def forward(self, hidden_states):
        # Extract the first token representation (CLS token)
        first_token = hidden_states[:, 0]
        transformed_output = self.linear_layer(first_token)
        activated_output = self.activation_fn(transformed_output)
        return activated_output


class ToxicityClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.weighting_factor = 0.8
        self.bert = BertModel(config, add_pooling_layer=False)
        self.pooler = CustomBertPooler(config)
        self.token_level_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.sequence_dropout = nn.Dropout(0.1)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, attn=None, labels=None):
        model_outputs = self.bert(input_ids, attention_mask)
        sequence_output = model_outputs[0]
        token_logits = self.token_classifier(self.token_level_dropout(sequence_output))

        pooled_representation = self.pooler(model_outputs[0])
        sequence_logits = self.sequence_classifier(self.sequence_dropout(pooled_representation))
        
        token_loss = None
        label_loss = None
        combined_loss = None

        if attn is not None:
            loss_function = nn.CrossEntropyLoss()
            # Handle masked tokens
            if mask is not None:
                active_positions = mask.view(-1) == 1
                active_token_logits = token_logits.view(-1, 2)
                active_token_labels = torch.where(
                    active_positions, attn.view(-1), torch.tensor(loss_function.ignore_index).type_as(attn)
                )
                token_loss = loss_function(active_token_logits, active_token_labels)
            else:
                token_loss = loss_function(token_logits.view(-1, 2), attn.view(-1))

            combined_loss = self.weighting_factor * token_loss

        if labels is not None:
            classification_loss_fn = nn.CrossEntropyLoss()
            label_loss = classification_loss_fn(sequence_logits.view(-1, self.num_labels), labels.view(-1))
            
            if combined_loss is not None:
                combined_loss += label_loss
            else:
                combined_loss = label_loss
                
        if combined_loss is not None:
            return sequence_logits, token_logits, combined_loss
        else:
            return sequence_logits, token_logits


class ToxicHateXplain():
    def __init__(self, model_path, max_length, batch_size, use_gpu=torch.cuda.is_available()):
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ToxicityClassifier.from_pretrained(model_path)
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def scoring(self, hypo):
        device = self.device
        text_samples = [text for _, text in hypo.items()]

        all_toxicity_scores = []
        for i in tqdm(range(0, len(text_samples), self.batch_size)):
            with torch.no_grad():
                batch_texts = text_samples[i:i + self.batch_size]
                encoded_inputs = self.tokenizer(
                    text=batch_texts, 
                    return_tensors='pt', 
                    truncation=True,
                    padding=True, 
                    max_length=self.max_length
                )
                
                input_ids = encoded_inputs['input_ids'].to(device)
                attention_mask = encoded_inputs['attention_mask'].to(device)
                
                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
                all_toxicity_scores.extend(probabilities[:, 1].tolist())

        return all_toxicity_scores