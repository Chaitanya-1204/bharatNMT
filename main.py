import os

# Prepare cache directories and environment variables for HF datasets
os.makedirs("cache", exist_ok=True)
os.environ["HF_HOME"] = "./cache"
os.environ["HF_DATASETS_CACHE"] = "./cache"


import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
import evaluate
import datetime
import math
import argparse
import sentencepiece as spm

from logger import log as log_message
from data import build_dataset , train_tokenizer
from tokenize_data import tokenize_dataset
from build_dataloaders import build_dataloaders
from model import get_model , count_parameters
from training_utils import train_one_epoch, validate_one_epoch, generate_sample_predictions, save_checkpoint

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
# Arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
args = parser.parse_args()




# Build dataset
log_message("Building dataset...", prefix="MAIN")
dataset = build_dataset(config)  

# Train tokenizer
log_message("Training tokenizer...", prefix="MAIN")
train_tokenizer(config)  

# Tokenize Data 
log_message("Tokenizing dataset...", prefix="MAIN")
tokenizer = tokenize_dataset(config, dataset)

# Build dataloaders
log_message("Building dataloaders...", prefix="MAIN")
train_loader, val_loader  = build_dataloaders(config)


# Training configurations
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["lr"]
WEIGHT_DECAY = config["training"]["weight_decay"]
PATIENCE = config["training"]["patience"]
MAX_LENGTH = config["training"]["max_length"]
DEVICE = config["training"]["device"]
num_workers = config["training"]["num_workers"]
TOKENIZER_MODEL_PATH = config["paths"]["tokenizer_model"]

num_training_steps = len(train_loader) * EPOCHS
WARMUP_STEPS = int(0.06 * num_training_steps) 

# Get model
log_message("Loading model..." , prefix="MAIN")
model = get_model(config)

# Set device
if DEVICE == "cuda:1":
    torch.cuda.set_device(1)
model = model.to(DEVICE)

# Log model parameters
count_parameters(model)


start_epoch = 0
best_val_loss = float("inf")

# Resume training if specified
if args.resume is not None and os.path.isfile(args.resume):
    
    log_message(f"Resuming training from checkpoint: {args.resume}", prefix="MAIN")
    
    checkpoint = torch.load(args.resume, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer_state = checkpoint.get("optimizer_state_dict", None)
    scheduler_state = checkpoint.get("scheduler_state_dict", None)
    
    if optimizer_state:
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer.load_state_dict(optimizer_state)
        
    if scheduler_state:
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps
        )
        scheduler.load_state_dict(scheduler_state)
        
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))




# ---------------- OPTIMIZER & SCHEDULER ----------------
if not (args.resume is not None and os.path.isfile(args.resume) and "optimizer_state" in locals() and "scheduler_state" in locals()):
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) 
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

# ---------------- METRIC ----------------


scaler = torch.amp.GradScaler()
epochs_no_improve = 0




for epoch in range(start_epoch, EPOCHS):
    log_message(f"Epoch {epoch+1}/{EPOCHS} started.", prefix="MAIN")
    avg_train_loss, train_perplexity = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE, scaler)
    val_metric = evaluate.load("sacrebleu")
    avg_val_loss, val_perplexity, bleu_score , avg_bert_score = validate_one_epoch(model, val_loader, DEVICE, tokenizer, MAX_LENGTH , val_metric)

    log_message(
    f"Epoch {epoch+1}\n"
    f"Train Loss: {avg_train_loss:.4f}\n"
    f"Train Perplexity: {train_perplexity:.2f}\n"
    f"Val Loss: {avg_val_loss:.4f}\n"
    f"Val Perplexity: {val_perplexity:.2f}\n"
    f"BLEU: {bleu_score:.2f}\n"
    f"BERTScore: {avg_bert_score:.4f}",
    prefix="MAIN"
)
    sample_metric = evaluate.load("sacrebleu")
    generate_sample_predictions(model, val_loader, DEVICE, tokenizer, MAX_LENGTH , sample_metric)

    model_dir = config["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, config["paths"]["best_model_name"])
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        torch.save(model.state_dict(), model_save_path)
        log_message("Best model saved.", prefix="MAIN")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            log_message("Early stopping triggered.", prefix="MAIN")
            break

    save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss , config)
