import os
import re
from datasets import load_dataset
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

from logger import log as log_msg


# Regex patterns to verify scripts for different languages
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")  # Covers Hindi and Marathi scripts
GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]+")    # Gujarati script range
EN_RE = re.compile(r"[A-Za-z]")                  # English letters


def valid_script(text, lang):
    """
        Check if the given text contains valid script characters for the specified language.

        Args:
            text (str): The text to validate.
            lang (str): Language code ('hi', 'mr', 'gu', 'en').

        Returns:
            bool: True if text contains valid script characters or digits, False otherwise.
    """
    # Allow if script characters OR numbers/symbols are present
    if lang in ["hi", "mr"]:
        return bool(DEVANAGARI_RE.search(text)) or any(char.isdigit() for char in text)

    elif lang == "gu":
        return bool(GUJARATI_RE.search(text)) or any(char.isdigit() for char in text)

    elif lang == "en":
        return bool(EN_RE.search(text)) or any(char.isdigit() for char in text)

    return True


def length_ratio_ok(src, tgt, max_ratio=3.0):
    """
        Check if the length ratio between source and target sentences is within acceptable bounds.

        Args:
            src (str): Source sentence.
            tgt (str): Target sentence.
            max_ratio (float): Maximum allowed length ratio.

        Returns:
            bool: True if length ratio is acceptable, False otherwise.
    """
    
    src_len = len(src.split())
    tgt_len = len(tgt.split())

    ratio = max(src_len / tgt_len, tgt_len / src_len)
    return ratio <= max_ratio


def length_limits_ok(src, tgt, min_len=2, max_len=128):
    """
        Check if both source and target sentences have lengths within specified limits.

        Args:
            src (str): Source sentence.
            tgt (str): Target sentence.
            min_len (int): Minimum acceptable length.
            max_len (int): Maximum acceptable length.

        Returns:
            bool: True if both sentences meet length constraints, False otherwise.
    """
    # Check that both sentences have token counts within acceptable bounds
    return (min_len <= len(src.split()) <= max_len) and (min_len <= len(tgt.split()) <= max_len)


def make_pairs(dataset, src_lang, tgt_lang):
    """
        Extract and filter parallel sentence pairs from the dataset.

        Args:
            dataset (DatasetDict): Dataset containing parallel sentences.
            src_lang (str): Source language code.
            tgt_lang (str): Target language code.

        Returns:
            pd.DataFrame: DataFrame containing filtered sentence pairs with columns [src_lang, tgt_lang].
    """
    pairs = []

    # Iterate over training data 
    for row in tqdm(dataset["train"], desc=f"Filtering {src_lang}→{tgt_lang}"):
        src = row["src"].strip()
        tgt = row["tgt"].strip()

        # Skip if either sentence is empty
        if not src or not tgt:
            continue

        # Skip pairs with extreme length mismatches
        if not length_ratio_ok(src, tgt):
            continue

        # Skip pairs that are too short or too long
        if not length_limits_ok(src, tgt):
            continue

        # Verify that the sentences contain the expected scripts
        if not valid_script(src, src_lang) or not valid_script(tgt, tgt_lang):
            continue

        pairs.append((src, tgt))

    # Convert list of tuples to DataFrame and remove duplicates
    df = pd.DataFrame(pairs, columns=[src_lang, tgt_lang])
    df = df.drop_duplicates()

    return df


def expand_bidirectional(df, lang1, lang2):
    """
        Generate bidirectional sentence pairs with language tags.

        Args:
            df (pd.DataFrame): DataFrame with columns [lang1, lang2].
            lang1 (str): First language code.
            lang2 (str): Second language code.

        Returns:
            pd.DataFrame: DataFrame containing bidirectional pairs with columns ['src_text', 'tgt_text', 'src_lang', 'tgt_lang'].
    """
    # Create two DataFrames: one for lang1→lang2, one for lang2→lang1 with language tags
    forward = pd.DataFrame({
        "src_text": [f"<2{lang2}> {x}" for x in df[lang1]],
        "tgt_text": df[lang2],
        "src_lang": lang1,
        "tgt_lang": lang2,
    })

    backward = pd.DataFrame({
        "src_text": [f"<2{lang1}> {x}" for x in df[lang2]],
        "tgt_text": df[lang1],
        "src_lang": lang2,
        "tgt_lang": lang1,
    })

    # Combine both directions into one DataFrame
    return pd.concat([forward, backward], ignore_index=True)


def build_dataset(config):
    """
        Build and save a merged dataset of filtered parallel sentence pairs for multiple languages.

        Args:
            config (dict): Configuration dictionary containing paths and dataset parameters.

        Returns:
            pd.DataFrame: Merged DataFrame containing all bidirectional sentence pairs.
    """
    output_dir = config["paths"]["data_dir"]
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, config["paths"]["dataset_file_name"])

    # If dataset already exists, skip processing
    if os.path.exists(output_path):
        log_msg(f"Dataset already exists at {output_path}. Skipping dataset build.", prefix="DATA")
        df = pd.read_csv(output_path, sep="\t")  # Load existing dataset
        return df

    log_msg("Loading datasets...", prefix="DATA")

    # Load datasets for Gujarati, Hindi, Marathi from Samanantar
    gu_ds = load_dataset("ai4bharat/samanantar", "gu")
    hi_ds = load_dataset("ai4bharat/samanantar", "hi")
    mr_ds = load_dataset("ai4bharat/samanantar", "mr")

    sentence_pairs = config["dataset"]["total_sentence_pairs"]

    # Shuffle and limit number of sentence pairs for each language dataset
    log_msg("Sampling gujarati datasets...", prefix="DATA")
    gu_ds["train"] = gu_ds["train"].shuffle(seed=42).select(range(min(sentence_pairs, len(gu_ds["train"]))))

    log_msg("Sampling hindi datasets...", prefix="DATA")
    hi_ds["train"] = hi_ds["train"].shuffle(seed=42).select(range(min(sentence_pairs, len(hi_ds["train"]))))

    log_msg("Sampling marathi datasets...", prefix="DATA")
    mr_ds["train"] = mr_ds["train"].shuffle(seed=42).select(range(min(sentence_pairs, len(mr_ds["train"]))))

    log_msg("Extracting and filtering sentence pairs...", prefix="DATA")

    # Extract filtered parallel pairs for each language pair
    gu_df = make_pairs(gu_ds, "en", "gu")
    hi_df = make_pairs(hi_ds, "en", "hi")
    mr_df = make_pairs(mr_ds, "en", "mr")

    log_msg("Generating bidirectional sentence pairs...", prefix="DATA")

    # Create bidirectional pairs with language tags
    pairs_gu = expand_bidirectional(gu_df, "en", "gu")
    pairs_hi = expand_bidirectional(hi_df, "en", "hi")
    pairs_mr = expand_bidirectional(mr_df, "en", "mr")

    log_msg("Merging datasets...", prefix="DATA")

    # Combine all language pairs into one DataFrame and shuffle
    all_pairs = pd.concat([pairs_gu, pairs_hi, pairs_mr], ignore_index=True)
    all_pairs = all_pairs.sample(frac=1).reset_index(drop=True)  # shuffle

    log_msg("Saving merged dataset...", prefix="DATA")

    # Save the final merged dataset as a TSV file
    all_pairs.to_csv(output_path, sep="\t", index=False)

    log_msg(f"Dataset saved to {output_path}", prefix="DATA")
    log_msg(f"Total sentence pairs: {len(all_pairs):,}", prefix="DATA")

    return all_pairs


def train_tokenizer(config):
    """
        Train a SentencePiece tokenizer on the merged dataset and save the tokenizer model files.

        Args:
            config (dict): Configuration dictionary containing paths and tokenizer parameters.
    """
    tokenizer_model_path = config["paths"]["tokenizer_model"]
    vocab_file_path = config["paths"]["vocab_file"]

    # If tokenizer files already exist, no need to retrain
    if os.path.exists(tokenizer_model_path) and os.path.exists(vocab_file_path):
        log_msg("Tokenizer files already exist. Skipping tokenizer training.", prefix="DATA")
        return

    tokenizer_folder = config["paths"]["tokenizer_folder"]
    dataset_path = os.path.join(config["paths"]["data_dir"], config["paths"]["dataset_file_name"])
    tokenizer_model_prefix = config["paths"]["tokenizer_model_prefix"]
    vocab_size = config["dataset"]["vocab_size"]
    model_type = config["dataset"]["model_type"]

    log_msg("Loading merged dataset for tokenizer training...", prefix="DATA")

    # Load the merged parallel dataset
    df = pd.read_csv(dataset_path, sep="\t")

    log_msg("Preparing corpus file for tokenizer training...", prefix="DATA")

    # Write all source and target sentences into a single corpus file for training
    corpus_path = "cache/total_corpus_final.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in tqdm(df["src_text"].tolist() + df["tgt_text"].tolist(), desc="Writing corpus"):
            f.write(text.strip() + "\n")

    os.makedirs(tokenizer_folder, exist_ok=True)

    log_msg("Training SentencePiece tokenizer...", prefix="DATA")

    # Train the SentencePiece tokenizer model with specified parameters
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=tokenizer_model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    os.remove(corpus_path)

    log_msg("Tokenizer training complete and saved.", prefix="DATA")
