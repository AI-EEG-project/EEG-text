"""
EEG-to-Text Benchmark Config Generator
Author: Lupse Ioan Victor (Person 3)

This script parses text/sentence files, computes word frequencies,
filters out common stopwords and punctuation, and generates a clean,
scaled-down benchmark_config.json file in the EXACT format required by
retrain_infonce.py and other training pipelines (including 'vocabulary',
'vocabulary_pos', and standard split lists).
"""

import os
import re
import json
import random
from collections import Counter

# Standard set of common English stopwords to avoid external dependencies (nltk/spacy)
STOPWORDS = {
    'the', 'and', 'of', 'to', 'in', 'is', 'that', 'it', 'on', 'you', 'he', 'was',
    'for', 'with', 'as', 'are', 'his', 'they', 'at', 'be', 'this', 'have', 'from',
    'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we',
    'when', 'your', 'can', 'said', 'there', 'use', 'an', 'each', 'which', 'she',
    'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many',
    'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'him',
    'into', 'time', 'has', 'look', 'two', 'more', 'write', 'go', 'see', 'no', 'way',
    'could', 'my', 'than', 'first', 'water', 'been', 'call', 'who', 'am', 'its',
    'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
}

# Predefined POS dictionary for common words to match standard annotations without Spacy
POS_MAPPING = {
    "born": "VERB", "american": "PROPN", "bush": "PROPN", "president": "NOUN",
    "new": "ADJ", "school": "NOUN", "film": "NOUN", "one": "NUM", "first": "ADV",
    "married": "VERB", "university": "NOUN", "movie": "NOUN", "united": "VERB",
    "states": "NOUN", "family": "NOUN", "john": "PROPN", "york": "PROPN",
    "college": "NOUN", "son": "NOUN", "january": "PROPN", "time": "NOUN",
    "company": "NOUN", "became": "VERB", "george": "PROPN", "years": "NOUN",
    "year": "NOUN", "best": "ADV", "kennedy": "PROPN", "actress": "NOUN",
    "july": "PROPN", "war": "NOUN", "ford": "PROPN", "later": "ADV",
    "november": "PROPN", "known": "VERB", "like": "INTJ", "served": "VERB",
    "two": "NUM", "april": "PROPN", "work": "VERB", "until": "ADP",
    "actor": "NOUN", "world": "NOUN", "august": "PROPN", "republican": "PROPN",
    "henry": "PROPN", "june": "PROPN", "roosevelt": "NOUN", "law": "NOUN",
    "member": "NOUN", "some": "PRON", "graduated": "VERB", "wife": "NOUN",
    "brother": "NOUN", "second": "ADJ", "william": "PROPN", "father": "PROPN",
    "city": "NOUN", "party": "NOUN", "television": "NOUN", "life": "NOUN",
    "state": "NOUN", "died": "VERB", "children": "NOUN", "october": "PROPN",
    "california": "PROPN", "elected": "VERB", "james": "PROPN", "attended": "VERB",
    "september": "PROPN", "house": "PROPN", "governor": "NOUN", "former": "ADJ",
    "director": "NOUN", "february": "PROPN", "december": "PROPN", "three": "NUM",
    "comedy": "NOUN", "career": "NOUN", "part": "NOUN", "role": "NOUN",
    "many": "ADJ", "movies": "NOUN", "made": "VERB", "age": "NOUN",
    "including": "VERB", "went": "VERB", "massachusetts": "PROPN", "march": "PROPN",
    "mother": "NOUN", "degree": "NOUN", "between": "ADP", "music": "NOUN",
    "national": "PROPN", "won": "VERB", "army": "NOUN", "england": "PROPN",
    "named": "VERB", "harvard": "PROPN", "charles": "PROPN", "following": "VERB",
    "become": "VERB", "often": "ADV", "great": "ADJ", "high": "ADJ",
    "left": "VERB", "under": "ADP", "moved": "VERB", "franklin": "PROPN",
    "received": "VERB", "democratic": "ADJ", "texas": "PROPN", "adams": "PROPN",
    "politician": "NOUN", "much": "ADJ", "british": "ADJ", "daughter": "PROPN",
    "king": "NOUN", "good": "ADJ", "another": "PRON", "little": "ADJ",
    "since": "SCONJ", "now": "ADV", "early": "ADV", "coppola": "PROPN",
    "clooney": "NOUN", "records": "NOUN", "secretary": "NOUN", "rockefeller": "PROPN",
    "island": "NOUN", "never": "ADV", "political": "ADJ", "serving": "VERB",
    "general": "ADJ", "ferrer": "NOUN", "yale": "PROPN", "simpson": "PROPN",
    "reagan": "PROPN", "young": "ADJ", "although": "SCONJ", "played": "VERB",
    "german": "NOUN", "presidential": "ADJ", "times": "NOUN", "robert": "PROPN",
    "last": "ADJ", "films": "NOUN", "series": "PROPN", "death": "NOUN",
    "senator": "PROPN", "election": "NOUN", "worked": "VERB", "academy": "PROPN",
    "defense": "NOUN", "funny": "ADJ", "picture": "NOUN", "again": "ADV",
    "long": "ADV", "characters": "NOUN", "home": "NOUN", "star": "PROPN",
    "award": "NOUN", "grant": "NOUN", "firm": "NOUN", "singer": "NOUN",
    "business": "NOUN", "child": "NOUN", "boston": "PROPN", "medal": "PROPN",
    "english": "PROPN", "rodham": "PROPN", "clinton": "PROPN", "barrymore": "NOUN",
    "acting": "VERB", "story": "NOUN", "love": "NOUN", "few": "ADJ",
    "despite": "SCONJ", "show": "VERB", "hollywood": "NOUN", "name": "NOUN",
    "once": "ADV", "senate": "PROPN", "entered": "VERB", "several": "ADJ",
    "man": "NOUN", "still": "ADV", "campaign": "NOUN", "famous": "ADJ",
    "founded": "VERB", "mary": "PROPN", "popular": "ADJ", "miguel": "PROPN",
    "educated": "VERB", "francis": "PROPN", "human": "NOUN", "old": "ADJ",
    "those": "PRON", "five": "NUM", "way": "NOUN"
}

def clean_text(text):
    """Lowercase text and strip punctuation."""
    text = text.lower()
    # Replace non-alphabetic characters with spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

def get_word_pos(word):
    """Returns the POS class of a word or defaults to NOUN if unknown."""
    return POS_MAPPING.get(word, "NOUN")

def build_smaller_benchmark(input_files, output_path="benchmark_config.json", target_vocab_size=50, min_len=3, seed=42):
    """
    Parses sentences from text, CSV, or MAT files, counts word frequencies,
    and splits the selected vocabulary into the exact unified format expected by models.
    """
    print(f"=== Starting Unified Benchmark Generator (Target Size: {target_vocab_size}) ===")

    random.seed(seed)
    word_counter = Counter()
    processed_sentences_count = 0
    fallback_used = False

    # 1. Gather Sentences from specified text/CSV/MAT files
    for file_path in input_files:
        if os.path.exists(file_path):
            print(f"Reading sentences from: {file_path}")
            try:
                ext = os.path.splitext(file_path)[1].lower()
                sentences_to_process = []

                # A. Handle CSV files
                if ext == '.csv':
                    import csv
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Auto-detect delimiter
                        sample = f.read(2048)
                        f.seek(0)
                        delimiter = ';' if ';' in sample else ','

                        reader = csv.reader(f, delimiter=delimiter)
                        header = next(reader, None)

                        sentence_col_idx = -1
                        if header:
                            header_lower = [h.strip().lower() for h in header]
                            for col_name in ['sentence', 'text', 'content', 'raw_text']:
                                if col_name in header_lower:
                                    sentence_col_idx = header_lower.index(col_name)
                                    break

                        for row in reader:
                            if not row:
                                continue
                            if sentence_col_idx != -1 and sentence_col_idx < len(row):
                                sentence = row[sentence_col_idx].strip()
                            else:
                                sentence = max(row, key=len).strip()
                            if len(sentence) > 5:
                                sentences_to_process.append(sentence)

                # B. Handle MATLAB files
                elif ext == '.mat':
                    import scipy.io as sio
                    import numpy as np

                    mat_contents = sio.loadmat(file_path, squeeze_me=True)

                    def extract_strings(data):
                        found = []
                        if isinstance(data, (str, bytes)):
                            s = data if isinstance(data, str) else data.decode('utf-8', errors='ignore')
                            s_clean = s.strip()
                            if len(s_clean) > 5 and ' ' in s_clean:
                                found.append(s_clean)
                        elif isinstance(data, np.ndarray):
                            if data.dtype.names is not None:
                                for name in data.dtype.names:
                                    found.extend(extract_strings(data[name]))
                            else:
                                if data.ndim == 0:
                                    found.extend(extract_strings(data.item()))
                                else:
                                    for item in data.flat:
                                        found.extend(extract_strings(item))
                        elif isinstance(data, dict):
                            for val in data.values():
                                found.extend(extract_strings(val))
                        elif isinstance(data, (list, tuple)):
                            for item in data:
                                found.extend(extract_strings(item))
                        return found

                    for key, val in mat_contents.items():
                        if not key.startswith('__'):
                            sentences_to_process.extend(extract_strings(val))

                # C. Handle standard pre-indexed or raw TXT files
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line_str = line.strip()
                            if not line_str:
                                continue

                            match = re.match(r'^\d+\s+(.*)$', line_str)
                            if match:
                                sentence = match.group(1)
                            else:
                                sentence = line_str

                            if len(sentence) > 5:
                                sentences_to_process.append(sentence)

                # Tokenize and filter vocabulary
                for sentence in sentences_to_process:
                    cleaned = clean_text(sentence)
                    words = cleaned.split()
                    filtered_words = [w for w in words if len(w) >= min_len and w not in STOPWORDS]
                    word_counter.update(filtered_words)
                    processed_sentences_count += 1

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # Fallback sentences
    if processed_sentences_count == 0:
        fallback_used = True
        print("No input files were successfully read. Using default fallback sentences.")
        fallback_sentences = [
            "The director made a highly emotional movie about the army during the war.",
            "This cinema is famous in Hollywood and has great actors in every film.",
            "An intelligent wife of a senator never worked in Hollywood before january.",
            "William went to Massachusetts in April to meet his daughter and family.",
            "Music is beautiful and often brings great joy to the actors during winter sessions."
        ]
        for sentence in fallback_sentences:
            cleaned = clean_text(sentence)
            words = cleaned.split()
            filtered_words = [w for w in words if len(w) >= min_len and w not in STOPWORDS]
            word_counter.update(filtered_words)
            processed_sentences_count += 1

    # Extract top words
    most_common = word_counter.most_common(target_vocab_size)
    selected_words = [word for word, count in most_common]

    actual_vocab_size = len(selected_words)
    if actual_vocab_size < target_vocab_size:
        print(f"Warning: Only {actual_vocab_size} words found.")
        target_vocab_size = actual_vocab_size

    # Sort the global vocabulary to keep matching indexes consistent
    selected_words = sorted(selected_words)

    # Deterministically Split
    # To keep train_words, val_words, and test_words disjoint, we split the shuffled vocabulary
    split_words = list(selected_words)
    random.shuffle(split_words)

    train_idx = int(actual_vocab_size * 0.70)
    val_idx = train_idx + int(actual_vocab_size * 0.15)

    train_words = sorted(split_words[:train_idx])
    val_words = sorted(split_words[train_idx:val_idx])
    test_words = sorted(split_words[val_idx:])

    # Generate POS mapping for the selected vocabulary
    vocabulary_pos = {word: get_word_pos(word) for word in selected_words}

    # Generate complete standard config payload
    config_payload = {
        "version": "1.0",
        "author": "Lupse Ioan Victor",
        "description": "Configuratie benchmark comuna EEG-to-Text scaled down",
        "seed": seed,
        "granularity": "word",
        "granularity_rationale": "Nivel cuvant: N400 definit per cuvant, mai multe exemple de antrenare",
        "vocab_size": actual_vocab_size,
        "vocab_selection": f"Top frecventa, excluse stopwords, min_len={min_len}, fallback: {fallback_used}",
        "split": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15
        },
        "corpus": {
            "custom_sentences": {
                "files": [os.path.basename(f) for f in input_files if os.path.exists(f)],
                "n": processed_sentences_count
            }
        },
        "eeg": {
            "n_channels": 105,
            "sampling_rate_hz": 500,
            "epoch_window_ms": [-200, 800],
            "baseline_ms": [-200, 0]
        },
        "vocabulary": selected_words,
        "vocabulary_pos": vocabulary_pos,
        "train_words": train_words,
        "val_words": val_words,
        "test_words": test_words,
        "evaluation_metrics": [
            "top1_accuracy",
            "top5_accuracy",
            "cosine_similarity_eeg_bert",
            "bleu1",
            "bleu2",
            "sentence_bert_similarity"
        ]
    }

    # Save to disk
    try:
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(config_payload, out, indent=2)
        print(f"Success! Benchmark config saved to: {output_path}")
        print(f"Master Vocab Size: {len(selected_words)}")
        print(f"Train/Val/Test split: {len(train_words)}/{len(val_words)}/{len(test_words)}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    import argparse as _ap

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    _parser = _ap.ArgumentParser(description="Generate benchmark_config.json for EEG-to-text training.")
    _parser.add_argument(
        "--size", type=int, default=200,
        help="Target vocabulary size (default: 200).",
    )
    _parser.add_argument(
        "--output", default=None,
        help=(
            "Output JSON path. Defaults to benchmark_config.json in repo root "
            "(or benchmark_config_50_Zuco1_SR.json when --size <= 50)."
        ),
    )
    _parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val/test split (default: 42).",
    )
    _args = _parser.parse_args()

    # Default output path: named after vocab size to avoid silently overwriting configs
    if _args.output:
        _out = _args.output
    elif _args.size <= 50:
        _out = os.path.join(ROOT_DIR, f"benchmark_config_{_args.size}_Zuco1_SR.json")
    else:
        _out = os.path.join(ROOT_DIR, "benchmark_config.json")

    _input_files = [
        os.path.join(ROOT_DIR, "data", "zuco1", "task1-SR", "preprocessed", "sentencesSR.mat"),
        os.path.join(ROOT_DIR, "data", "zuco1", "task1-SR", "preprocessed", "sentiment_normal_reading.csv"),
        os.path.join(ROOT_DIR, "data", "zuco1", "task2-NR", "preprocessed", "relations_normal_reading.csv"),
        os.path.join(ROOT_DIR, "data", "zuco1", "task2-NR", "preprocessed", "sentencesNR.mat"),
    ]

    build_smaller_benchmark(
        input_files=_input_files,
        output_path=_out,
        target_vocab_size=_args.size,
        seed=_args.seed,
    )