from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch

from logger import log as log_message

PAD_ID = 0
MAX_LEN = 256


def collate_fn(batch):
    """
        Collate function to pad or truncate sequences in a batch to a fixed length.

        Args:
            batch (list of dict): A batch of samples, each containing 'src_ids', 'tgt_input_ids', and 'tgt_labels'.

        Returns:
            dict: A dictionary with padded/truncated tensors for source ids, source mask, target input ids, and target labels.
    """
    def pad_or_truncate(seq_list, pad_id = PAD_ID):
        fixed = []
        for seq in seq_list:
            if len(seq) > MAX_LEN:
                fixed.append(seq[:MAX_LEN])
            else:
                fixed.append(seq + [pad_id] * (MAX_LEN - len(seq)))
        return torch.tensor(fixed)

    # Pad or truncate source sequences
    src = pad_or_truncate([x["src_ids"] for x in batch])

    # Pad or truncate target input sequences
    tgt_in = pad_or_truncate([x["tgt_input_ids"] for x in batch])

    # Pad or truncate target label sequences
    tgt_labels = pad_or_truncate([x["tgt_labels"] for x in batch])

    # Create attention mask for source (1 for tokens, 0 for padding)
    src_mask = (src != PAD_ID).long()

    return {
        "src_ids": src,
        "src_mask": src_mask,
        "tgt_input_ids": tgt_in,
        "tgt_labels": tgt_labels,
    }


def build_dataloaders(config):
    
    """
        Build PyTorch DataLoaders for training and validation datasets from a tokenized dataset.

        Args:
            config (dict): Configuration dictionary containing training parameters and dataset paths.

        Returns:
            tuple: A tuple containing the training DataLoader and validation DataLoader.
    """
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    # Load the tokenized dataset from disk
    log_message("Loading tokenized dataset...", prefix = "DATA_LOADERS")
    dataset = load_from_disk(config["paths"]["tokenized_data_folder"])

    # Split the dataset into training and validation sets (1% for validation)
    dataset_split = dataset.train_test_split(test_size = 0.01, seed = 42)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    log_message(f"Train size: {len(train_dataset)}", prefix = "DATA_LOADERS")
    log_message(f"Validation size: {len(val_dataset)}", prefix = "DATA_LOADERS")

    # Create DataLoaders for training and validation datasets
    log_message("Creating DataLoaders...", prefix = "DATA_LOADERS")
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_fn,
        num_workers = num_workers,
    )

    return train_loader, val_loader