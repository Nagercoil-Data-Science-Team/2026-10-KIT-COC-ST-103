"""
Dataset Preprocessor for Automated Essay Scoring (ASAP-AES Benchmark)
Strictly utilizes the authentic Kaggle ASAP-AES dataset without synthetic generation.
Implements Step 1 (Dataset Organization) and Step 2 (Text Preprocessing)
from the Workflow document.
"""

import os
import re
import unicodedata
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from sklearn.model_selection import train_test_split

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

# Score ranges defined in Kaggle ASAP-AES dataset for prompts 1 through 8
PROMPT_SCORE_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}


def load_real_asap_dataset(file_path=None, sample_size=None, random_state=42):
    """
    Loads authentic ASAP-AES dataset directly from local TSV/CSV files.
    Strictly forbids synthetic text generation.
    """
    candidates = [
        file_path,
        os.path.join(os.path.dirname(__file__), 'dataset', 'training_set_rel3.tsv'),
        os.path.join('dataset', 'training_set_rel3.tsv'),
        'training_set_rel3.tsv',
        'training_set_rel3.csv'
    ]

    df = None
    loaded_from = None
    for cand in candidates:
        if cand and os.path.exists(cand) and os.path.getsize(cand) > 100000:
            try:
                sep = '\t' if cand.endswith('.tsv') else ','
                # Read authentic ASAP-AES file
                df = pd.read_csv(cand, sep=sep, encoding='latin-1', on_bad_lines='skip')
                if 'essay' in df.columns and 'domain1_score' in df.columns and 'essay_set' in df.columns:
                    loaded_from = cand
                    break
            except Exception as e:
                print(f"[Dataset] Notice: Could not read {cand}: {e}")

    if df is None:
        raise FileNotFoundError(
            "Authentic ASAP-AES dataset not found! Please ensure 'training_set_rel3.tsv' "
            "is present in the 'dataset/' folder or project root."
        )

    print(f"[Dataset] Successfully loaded authentic ASAP dataset from: {loaded_from}")
    print(f"[Dataset] Total raw essays loaded: {len(df)}")

    # If stratified sampling is requested for fast CPU benchmarking
    if sample_size is not None and sample_size < len(df):
        print(f"[Dataset] Performing stratified sample of {sample_size} essays across prompts 1-8...")
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=random_state,
            stratify=df['essay_set']
        )
        df = df.reset_index(drop=True)
        print(f"[Dataset] Stratified subset created: {len(df)} essays.")

    return df


class ASAPTextPreprocessor:
    """
    Implements Step 2 Text Preprocessing:
    - Missing-value handling (median imputation for score)
    - Duplicate removal
    - HTML/tag removal (BeautifulSoup)
    - Special-character cleaning (regex)
    - Whitespace normalization
    - Text normalization (Unicode NFKD, lowercasing)
    - Score validation (within official PROMPT_SCORE_RANGES)
    - Sentence segmentation (NLTK)
    - Score min-max normalization per prompt set for regression training
    """
    def __init__(self):
        pass

    def clean_text(self, text):
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # HTML tag removal
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Special character cleaning (keep standard essay punctuation and alphanumeric)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"\-]', ' ', text)
        
        # Whitespace normalization
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Text normalization: lowercasing
        text = text.lower()
        return text

    def segment_sentences(self, text):
        if not text:
            return []
        return sent_tokenize(text)

    def preprocess_dataset(self, df):
        print("[Preprocessing] Step 2: Executing full text preprocessing pipeline...")
        df_clean = df.copy()
        
        # 1. Missing-value handling
        initial_count = len(df_clean)
        df_clean['essay'] = df_clean['essay'].fillna("")
        df_clean['domain1_score'] = pd.to_numeric(df_clean['domain1_score'], errors='coerce')
        df_clean['domain1_score'] = df_clean['domain1_score'].fillna(df_clean['domain1_score'].median())
        df_clean['essay_set'] = pd.to_numeric(df_clean['essay_set'], errors='coerce').astype(int)
        
        # 2. Duplicate removal
        df_clean = df_clean.drop_duplicates(subset=['essay']).reset_index(drop=True)
        num_dropped = initial_count - len(df_clean)
        if num_dropped > 0:
            print(f"[Preprocessing] Removed {num_dropped} duplicate essays.")
        
        # 3. Text cleaning
        df_clean['cleaned_essay'] = df_clean['essay'].apply(self.clean_text)
        
        # 4. Sentence segmentation
        df_clean['sentences'] = df_clean['cleaned_essay'].apply(self.segment_sentences)
        df_clean['num_sentences'] = df_clean['sentences'].apply(len)
        df_clean['word_count'] = df_clean['cleaned_essay'].apply(lambda s: len(s.split()))
        
        # Filter out empty or un-parsable essays
        df_clean = df_clean[df_clean['word_count'] > 3].reset_index(drop=True)
        
        # 5. Score validation
        valid_rows = []
        for idx, row in df_clean.iterrows():
            pset = int(row['essay_set'])
            s_min, s_max = PROMPT_SCORE_RANGES.get(pset, (0, 100))
            score = float(row['domain1_score'])
            if s_min <= score <= s_max:
                valid_rows.append(True)
            else:
                valid_rows.append(False)
        df_clean = df_clean[valid_rows].reset_index(drop=True)
        
        # 6. Normalized score (0.0 to 1.0) per essay set for model regression training
        df_clean['normalized_score'] = 0.0
        for pset, (s_min, s_max) in PROMPT_SCORE_RANGES.items():
            mask = df_clean['essay_set'] == pset
            if mask.sum() > 0:
                span = max(1, s_max - s_min)
                df_clean.loc[mask, 'normalized_score'] = (df_clean.loc[mask, 'domain1_score'] - s_min) / span

        print(f"[Preprocessing] Preprocessing complete. Valid cleaned essays: {len(df_clean)}")
        return df_clean


def get_dataset_splits(df, test_size=0.20, val_size=0.15, random_state=42):
    """
    Creates stratified Train / Validation / Test splits preserving essay_set distribution.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['essay_set']
    )
    # Remaining train split into train and validation
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=random_state, stratify=train_df['essay_set']
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Backwards compatibility alias
load_or_create_asap_dataset = load_real_asap_dataset


if __name__ == '__main__':
    df = load_real_asap_dataset()
    preprocessor = ASAPTextPreprocessor()
    df_clean = preprocessor.preprocess_dataset(df)
    train_df, val_df, test_df = get_dataset_splits(df_clean)
    print(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("\nSample processed records:")
    print(df_clean[['essay_id', 'essay_set', 'domain1_score', 'normalized_score', 'word_count', 'num_sentences']].head())
