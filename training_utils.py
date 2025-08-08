import torch
from logger import log as log_message
from tqdm import tqdm
import math   
import bert_score
import os

def train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler):
    """
        Runs one epoch of training.

        Args:
            model: The NMT model to train.
            train_loader: DataLoader providing training batches.
            optimizer: Optimizer for updating model parameters.
            scheduler: Learning rate scheduler.
            device: Device to run the computations on (e.g., "cuda:0").
            scaler: GradScaler for mixed precision training.

        Returns:
            avg_train_loss (float): Average loss over the epoch.
            train_perplexity (float): Perplexity computed from the average loss.
    """
    
    model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        # Move batch tensors to the specified device (CPU or GPU)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch.float16):
            
            outputs = model(
                input_ids=batch["src_ids"],
                attention_mask=batch["src_mask"],
                decoder_input_ids=batch["tgt_input_ids"],
                labels=batch["tgt_labels"]
            )
            
            loss = outputs.loss

        # Backward pass and optimization step with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accumulate training loss
        total_train_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss:.4f}"})

    # Compute average loss and perplexity for the epoch
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_perplexity = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
    
    return avg_train_loss, train_perplexity



def validate_one_epoch(model, val_loader, device, tokenizer, max_length, metric):
    """
        Runs one epoch of validation for the NMT model.

        Args:
            model: The NMT model to validate.
            val_loader: DataLoader providing validation batches.
            device: Device to run the computations on (e.g., "cuda:0").
            tokenizer: Tokenizer for decoding predictions and labels.
            max_length (int): Maximum sequence length for generation.
            metric: Metric object (e.g., BLEU) for evaluation.

        Returns:
            avg_val_loss (float): Average validation loss over the epoch.
            val_perplexity (float): Perplexity computed from the average validation loss.
            bleu_score (float): BLEU score for the validation set.
            avg_bert_score (float): Average BERTScore F1 for the validation set.
    """
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc="Validation")
        for batch in val_progress_bar:
            # Move batch tensors to the specified device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch.float16):
                
                outputs = model(
                    input_ids=batch["src_ids"],
                    attention_mask=batch["src_mask"],
                    decoder_input_ids=batch["tgt_input_ids"],
                    labels=batch["tgt_labels"]
                )
                
                loss = outputs.loss
                
                total_val_loss += loss.item()
                avg_val_loss_so_far = total_val_loss / (len(val_progress_bar) if len(val_progress_bar) > 0 else 1)
                val_progress_bar.set_postfix({"val_loss": f"{avg_val_loss_so_far:.4f}"})

            # Generate predictions for the batch
            generated_tokens = model.generate(batch["src_ids"], max_length=max_length)
            decoded_preds = [tokenizer.decode(ids.tolist()) for ids in generated_tokens]
            decoded_labels = [tokenizer.decode(ids.tolist()) for ids in batch["tgt_labels"]]

            # Add batch predictions and references to the metric for BLEU calculation
            metric.add_batch(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    # Compute average validation loss and perplexity
    avg_val_loss = total_val_loss / len(val_loader)
    val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")

    # Compute BLEU score for the validation set
    bleu_score = metric.compute()["score"]

    # Compute BERTScore for the validation set
    bert_score_result = bert_score.score(decoded_preds, decoded_labels, lang="en", verbose=False)
    avg_bert_score = bert_score_result[2].mean().item()

    return avg_val_loss, val_perplexity, bleu_score, avg_bert_score



def generate_sample_predictions(model, val_loader, device, tokenizer, max_length, sample_bleu):
    """
        Generates and logs sample predictions from the validation set for qualitative assessment.

        Args:
            model: The NMT model for generating predictions.
            val_loader: DataLoader providing validation batches.
            device: Device to run the computations 
            tokenizer: Tokenizer for decoding predictions and labels.
            max_length (int): Maximum sequence length for generation.
            metric: Metric object (not directly used here, but for consistency).

        Returns:
            None
    """
    log_message("Sample Predictions:", prefix="TRAINING")
    sample_count = 0
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            # Move batch tensors to the specified device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Generate translations using beam search and constraints
            generated_tokens = model.generate(
                batch["src_ids"],
                attention_mask=batch["src_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                eos_token_id=tokenizer.eos_id(),
                bos_token_id=tokenizer.bos_id()
            )

            # Decode predictions and true labels to text
            decoded_preds = [tokenizer.decode(ids.tolist()) for ids in generated_tokens]
            decoded_labels = [tokenizer.decode(ids.tolist()) for ids in batch["tgt_labels"]]

            # Log  10 sample predictions, true labels, and BLEU scores
            for src, pred, tgt in zip(batch["src_ids"], decoded_preds, decoded_labels):
                src_text = tokenizer.decode(src.tolist())
                log_message(f"SRC: {src_text}", prefix="TRAINING")
                log_message(f"PRED: {pred}", prefix="TRAINING")
                log_message(f"TRUE: {tgt}", prefix="TRAINING")

                # Compute BLEU score for the sample
                
                sample_bleu.add_batch(predictions=[pred], references=[[tgt]])
                sample_bleu_score = sample_bleu.compute()["score"]
                log_message(f"SAMPLE BLEU: {sample_bleu_score:.2f}", prefix="TRAINING")

                # Compute BERTScore for the sample
                _, _, sample_bert_f1 = bert_score.score([pred], [tgt], lang="en", verbose=False)
                log_message(f"SAMPLE BERTScore: {sample_bert_f1[0].item():.4f}", prefix="TRAINING")

                sample_count += 1
                if sample_count >= 10:
                    break
            if sample_count >= 10:
                break



def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss , config):
    """
    Saves the current model, optimizer, and scheduler state to a checkpoint file.

    Args:
        model: The NMT model to save.
        optimizer: Optimizer whose state to save.
        scheduler: Scheduler whose state to save.
        epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss achieved so far.

    Returns:
        None
    """
    
    checkpoint_diectory = config["paths"]["output_dir"]
    os.makedirs(checkpoint_diectory, exist_ok=True)
        
    checkpoint_path = config["paths"]["checkpoint_paths"].format(epoch=epoch)
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss
    }, checkpoint_path)
    
    log_message(f"Checkpoint saved: {checkpoint_path}", prefix="TRAINING")
