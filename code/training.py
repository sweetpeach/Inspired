
import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, WarmupLinearSchedule, GPT2Config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "CUDA_INDEX"
torch.cuda.manual_seed_all(0)
np.random.seed(0)

class InspiredDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = [628, 198]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        return role_ids, dial_tokens
        
    def collate(self, unpacked_data):
        return unpacked_data

# Define loss
class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask

    if reduce:
        # shape : (batch,)
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce is "batch":
            # shape : scalar
            loss = loss.mean()

    return loss

# Training
def train_one_iter(batch, update_count, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    
    past = None
    all_logits = []
    
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            logits, past = model_A(dial_turn_inputs, past=past)
            all_logits.append(logits)
        else:
            logits, past = model_B(dial_turn_inputs, past=past)
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=1)
    
    # target
    all_logits = all_logits[:, :-1].contiguous()
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
    target_mask = torch.ones_like(target).float()
    
    loss = criterion(all_logits, target, target_mask, label_smoothing=0.02, reduce="batch")   
    loss /= num_gradients_accumulation

    loss.backward()
        
    record_loss = loss.item() * num_gradients_accumulation
    perplexity = np.exp(record_loss)
    
    return record_loss, perplexity


def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        total_ppl = []
        total_ppl_recommender = []

        for batch in pbar:
            if sum([len(item) for item in batch[0][1]]) > 1024:
                total_length = 0
                for index, item in enumerate(batch[0][1]):
                    total_length = total_length + len(item)
                    if total_length >= 1024:
                        batch = [(batch[0][0][0:index-1], batch[0][1][0:index-1])]
                        break
        
            role_ids, dialog_tokens = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
            dial_inputs_rec = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens if item[0] == 32]
            past = None
            all_logits = []
            all_logits_rec = []

            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    logits, past = model_A(dial_turn_inputs, past=past)
                    all_logits.append(logits)
                    
                    all_logits_rec.append(logits)
                else:
                    logits, past = model_B(dial_turn_inputs, past=past)
                    all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=1)
            all_logits_rec = torch.cat(all_logits_rec, dim=1)
            
            # target
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()
            
            # target of rec
            all_logits_rec = all_logits_rec[:, :-1].contiguous()
            target_rec = torch.cat(dial_inputs_rec, dim=1)[:, 1:].contiguous()
            target_mask_rec = torch.ones_like(target_rec).float()
            
            loss = criterion(all_logits, target, target_mask, label_smoothing=-1, reduce="sentence")      
            loss_recommender = criterion(all_logits_rec, target_rec, target_mask_rec, label_smoothing = -1, reduce="sentence")
            
            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())
            
            ppl_recommender = torch.exp(loss_recommender)
            total_ppl_recommender.extend(ppl_recommender.tolist())

        print(f"Epcoh {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        print(f"Epcoh {ep} Validation Perplexity on recommender (A): {np.mean(total_ppl_recommender)} Variance: {np.var(total_ppl_recommender)}")
        
        return np.mean(total_ppl)


# inspired_data
inspired_data = torch.load("DATA_PATH")
data = inspired_data
indices = np.arange(len(data))
np.random.shuffle(indices)

train_data = [data[idx] for idx in indices[int(0.2*len(data)):]]
val_data = [data[idx] for idx in indices[:int(0.1*len(data))]]
test_data = [data[idx] for idx in indices[int(0.1*len(data)):int(0.2*len(data))]]

batch_size = 1
tokenizer = torch.load("TOKENIZER_PATH")

train_dataset = InspiredDataset(train_data, tokenizer)
val_dataset = InspiredDataset(val_data, tokenizer)
train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=train_dataset.collate)
val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)

# continue training
# Initialize models
model_A_states, model_B_states = torch.load("PRETRAINING_MODEL_WEIGHT")

added_embedding_plot = [model_A_states["transformer.wte.weight"][[tokenizer.encode("movie plot")], :].mean(1)]
added_embedding_plot = torch.cat(added_embedding_plot, 0)
added_embedding_sep = [model_A_states["transformer.wte.weight"][[tokenizer.encode("separate")], :].mean(1)]
added_embedding_sep = torch.cat(added_embedding_sep, 0)

new_weight = torch.cat([model_A_states["transformer.wte.weight"], added_embedding_plot, added_embedding_sep], 0)

model_A_states["transformer.wte.weight"] = new_weight
model_B_states["transformer.wte.weight"] = new_weight
model_A_states["lm_head.weight"] = new_weight
model_B_states["lm_head.weight"] = new_weight

config = GPT2Config()
config.vocab_size = model_A_states["transformer.wte.weight"].shape[0]
model_A = GPT2LMHeadModel(config)
model_B = GPT2LMHeadModel(config)

model_A.load_state_dict(model_A_states)
model_B.load_state_dict(model_B_states)

device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)

criterion = SequenceCrossEntropyLoss()

# Optimizer
# define hyper-parameters
num_epochs = 10
num_gradients_accumulation = 1
num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
no_decay = ['bias', 'ln', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)

scheduler = WarmupLinearSchedule(optimizer,
                                 warmup_steps=100,
                                 t_total=num_train_optimization_steps)

update_count = 0
progress_bar = tqdm.tqdm_notebook
start = time.time()
old_ppl = -float('Inf')

for ep in range(num_epochs):

    #"Training"
    pbar = progress_bar(train_dataloader)
    model_A.train()
    model_B.train()
    
    for batch in pbar:
        batch = batch[0]
        
        if sum([len(item) for item in batch[1]]) > 1024:
            total_length = 0
            for index, item in enumerate(batch[1]):
                total_length = total_length + len(item)
                if total_length >= 1024:
                    batch = (batch[0][0:index-1], batch[1][0:index-1])
                    break
    
        record_loss, perplexity = train_one_iter(batch, update_count, fp16=False)
        update_count += 1

        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            # update for gradient accumulation
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end
            
            # show progress
            pbar.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)

    #"Evaluation"
    model_A.eval()
    model_B.eval()
    ppl = validate(val_dataloader)

    # save the model for later use
    torch.save([model_A.state_dict(), model_B.state_dict()], f"SAVE_MODEL_PATH")


