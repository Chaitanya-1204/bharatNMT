import pandas as pd
import sentencepiece as spm
from datasets import Dataset
import os
from tqdm import tqdm
from logger import log as log_msg

# Encoding functions
def encode_src(text, sp):
    
    """
        Encode source text using SentencePiece tokenizer.

        Args:
            text (str): The source text to encode.
            sp (SentencePieceProcessor): The SentencePiece tokenizer instance.

        Returns:
            List[int]: List of token IDs representing the encoded source text.
    """
    
    return sp.encode(text, out_type=int)

def encode_tgt(text, sp):
    
    """
        Create decoder input and label sequences with BOS and EOS tokens.

        Args:
            text (str): The target text to encode.
            sp (SentencePieceProcessor): The SentencePiece tokenizer instance.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - decoder_input: token IDs starting with BOS token.
                - labels: token IDs ending with EOS token.
    """
    
    tokens = sp.encode(text, out_type=int)
    decoder_input = [sp.bos_id()] + tokens
    labels = tokens + [sp.eos_id()]
    
    return decoder_input, labels

# Tokenization function
def tokenize_dataset(config , df):
    
    """
        Load dataset, tokenize source and target texts, and save as Hugging Face Dataset.

        Args:
            config (dict): Configuration dictionary containing paths for data, tokenizer model,
                        and output directory.
            df (pd.DataFrame): DataFrame containing 'src_text' and 'tgt_text' columns.

        Process:
            - Loads the raw dataset from a CSV file.
            - Loads the SentencePiece tokenizer model.
            - Encodes source texts and target texts with proper BOS/EOS tokens.
            - Stores tokenized sequences in new columns.
            - Converts the DataFrame to a Hugging Face Dataset.
            - Saves the tokenized dataset to disk.
    """
    
    # Paths
    
    TOKENIZER_MODEL_PATH = config["paths"]["tokenizer_model"]
    OUTPUT_DIR = config["paths"]["tokenized_data_folder"]
    
    

    # Load tokenizer
    log_msg("Loading the tokenizer model...", prefix="TOKENIZE_DATA")
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL_PATH)
    log_msg("Tokenizer model has been loaded successfully.", prefix="TOKENIZE_DATA")

    # Applying encoding
    log_msg("Starting to tokenize source and target texts...", prefix="TOKENIZE_DATA")
    src_ids_list = []
    tgt_in_ids_list = []
    tgt_labels_list = []

    for src, tgt in tqdm(zip(df["src_text"], df["tgt_text"]), total=len(df), desc="Tokenizing dataset"):
        src_ids = encode_src(src, sp)
        decoder_in, labels = encode_tgt(tgt, sp)
        src_ids_list.append(src_ids)
        tgt_in_ids_list.append(decoder_in)
        tgt_labels_list.append(labels)

    log_msg("Finished tokenizing all dataset samples.", prefix="TOKENIZE_DATA")


    df["src_ids"] = src_ids_list
    df["tgt_input_ids"] = tgt_in_ids_list
    df["tgt_labels"] = tgt_labels_list

    # Convert to Hugging Face Dataset for fast loading
    log_msg("Saving the tokenized dataset in Hugging Face format...", prefix="TOKENIZE_DATA")
    hf_dataset = Dataset.from_pandas(df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_dataset.save_to_disk(OUTPUT_DIR)


    log_msg(f"Tokenized dataset saved successfully at {OUTPUT_DIR}", prefix="TOKENIZE_DATA")
    log_msg(f"Total number of samples processed: {len(hf_dataset)}", prefix="TOKENIZE_DATA")
    
    return sp