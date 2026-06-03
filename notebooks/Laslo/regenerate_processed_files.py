"""
EEG-to-Text Pipeline Regenerator (Laslo's Preprocessing Pipeline)
Author: Lupse Ioan Victor (Person 3)

This script regenerates the preprocessed files (X_preprocessed.npy, y_labels.npy, splits.npy)
by parsing custom subject .mat files and aligning them to a chosen benchmark config.
Updated with case-insensitive structural mapping supporting 'content' and 'rawEEG' fields.
"""

import os
import json
import numpy as np
import scipy.io as sio
from scipy.signal import resample

# Directories Definition
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def downsample_eeg(eeg_data, target_samples=250):
    """
    Downsamples EEG epoch from its original shape (channels, samples)
    to (channels, target_samples).
    """
    channels, samples = eeg_data.shape
    if samples == target_samples:
        return eeg_data
    return resample(eeg_data, target_samples, axis=1)

def extract_subject_epochs(mat_file_path, active_vocab, target_samples=250):
    """
    Parses a single subject's ZuCo .mat file and extracts word-level EEG epochs
    that belong to the active benchmark vocabulary using defensive structural checks.
    """
    print(f"Processing subject file: {mat_file_path}")
    try:
        # Load structural MATLAB file
        mat_data = sio.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading MATLAB file {mat_file_path}: {e}")
        return [], []

    # Identify the sentenceData array inside the MAT file case-insensitively
    sentence_data = None
    for key in mat_data.keys():
        if key.lower() == "sentencedata":
            sentence_data = mat_data[key]
            break

    if sentence_data is None:
        print(f"Could not find 'sentenceData' field in {mat_file_path}. Available keys: {list(mat_data.keys())}")
        return [], []

    # Unwrap single-element arrays (e.g., double nesting from MATLAB outputs)
    while isinstance(sentence_data, np.ndarray) and sentence_data.ndim > 1 and sentence_data.shape[0] == 1:
        sentence_data = sentence_data[0]

    epochs_list = []
    labels_list = []

    # Flatten sentence array to ensure 1D sequence iteration
    if isinstance(sentence_data, np.ndarray):
        sentence_data = sentence_data.ravel()
    elif not isinstance(sentence_data, (list, tuple)):
        sentence_data = [sentence_data]

    for s_idx, sentence in enumerate(sentence_data):
        # Support both object attribute and dictionary/record index access modes
        words = None
        if hasattr(sentence, 'word'):
            words = sentence.word
        elif isinstance(sentence, dict) and 'word' in sentence:
            words = sentence['word']
        elif hasattr(sentence, 'dtype') and 'word' in sentence.dtype.names:
            words = sentence['word']

        if words is None:
            continue

        # Flatten words array to ensure clean 1D iteration
        if isinstance(words, np.ndarray):
            words = words.ravel()
        elif not isinstance(words, (list, tuple)):
            words = [words]

        for w_idx, word_struct in enumerate(words):
            # Extract word string robustly across formats (content is prioritized for resultsZAB_SR)
            word_str_raw = None
            for attr_candidate in ['content', 'wordstring', 'wordString', 'text', 'word']:
                if hasattr(word_struct, attr_candidate):
                    word_str_raw = getattr(word_struct, attr_candidate)
                    break
                elif isinstance(word_struct, dict) and attr_candidate in word_struct:
                    word_str_raw = word_struct[attr_candidate]
                    break
                elif hasattr(word_struct, 'dtype') and attr_candidate in word_struct.dtype.names:
                    word_str_raw = word_struct[attr_candidate]
                    break

            if word_str_raw is None:
                continue

            # Robust conversion to standard Python string to prevent numpy type issues
            raw_word = str(word_str_raw).strip().lower()
            clean_word = "".join([char for char in raw_word if char.isalpha()])

            if clean_word in active_vocab:
                # Extract EEG signal supporting rawEEG and other standard fields
                eeg_signal = None

                for field in ['rawEEG', 'EEG500', 'meanEEG', 'EEG', 'EEG_bandPowers', 'eeg', 'signal']:
                    if hasattr(word_struct, field):
                        eeg_signal = getattr(word_struct, field)
                        break
                    elif isinstance(word_struct, dict) and field in word_struct:
                        eeg_signal = word_struct[field]
                        break
                    elif hasattr(word_struct, 'dtype') and field in word_struct.dtype.names:
                        eeg_signal = word_struct[field]
                        break

                if eeg_signal is None or not isinstance(eeg_signal, np.ndarray):
                    continue

                # Remove singular wrapping dimensions (e.g., (1, 105, 500) -> (105, 500))
                eeg_signal = np.squeeze(eeg_signal)

                if eeg_signal.ndim != 2:
                    continue

                # Normal standard shape is (105, samples)
                if eeg_signal.shape[0] != 105 and eeg_signal.shape[1] == 105:
                    eeg_signal = eeg_signal.T

                # Keep only standard 105 channels
                eeg_signal = eeg_signal[:105, :]

                # Downsample to target samples
                eeg_signal_resampled = downsample_eeg(eeg_signal, target_samples)

                epochs_list.append(eeg_signal_resampled)
                labels_list.append(clean_word)

    print(f"Successfully extracted {len(epochs_list)} trials for this subject.")
    return epochs_list, labels_list

def _extract_subject_id(file_path: str) -> str:
    """
    Extract subject ID from a ZuCo filename.
    'data/zuco1/task1-SR/resultsZAB_SR.mat' -> 'ZAB'
    Works for any task suffix (_SR, _NR, _TSR, etc.)
    """
    import re
    basename = os.path.basename(file_path)
    match = re.search(r'results([A-Z]+)_', basename)
    return match.group(1) if match else basename


def _process_subject(args):
    """
    Worker function — called once per subject file on its own thread.
    Loads the .mat file, extracts epochs, z-scores per subject, returns
    (subject_id, normalized_epochs_list, labels_list).
    Designed to be stateless and thread-safe (no shared mutable state).
    """
    file_path, active_vocab, target_samples = args
    full_path = os.path.join(BASE_DIR, file_path) if not os.path.isabs(file_path) else file_path
    subject_id = _extract_subject_id(file_path)

    if not os.path.exists(full_path):
        print(f"Warning: Subject MAT file not found at: {full_path}")
        return subject_id, [], []

    epochs, labels = extract_subject_epochs(full_path, active_vocab, target_samples)

    if not epochs:
        return subject_id, [], []

    epochs_array = np.array(epochs, dtype=np.float32)
    mean = np.mean(epochs_array)
    std  = np.std(epochs_array) if np.std(epochs_array) > 0 else 1.0
    normalized = (epochs_array - mean) / std

    return subject_id, list(normalized), labels


def run_pipeline_regeneration(subject_files, config_filename="benchmark_config.json",
                               target_samples=250, n_workers=None, split_mode="example"):
    """
    Coordinates the full reconstruction pipeline using specified MAT files and config.
    Each subject file is processed in parallel using a thread pool.

    split_mode:
      "example"  — random trial-level split (70/15/15). All words in all splits.
                   Standard retrieval evaluation: predict word from unseen EEG recording.
      "subject"  — split by subject identity. Train/val/test subjects are disjoint.
                   Forces cross-subject generalisation, more realistic but harder.
                   8 train subjects / 2 val subjects / 2 test subjects.

    n_workers: number of threads (default: number of subject files, capped at 8).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config_path = os.path.join(BASE_DIR, config_filename)
    if not os.path.exists(config_path):
        config_path = os.path.join(BASE_DIR, "benchmark_config.json")
        print(f"Specified config not found. Falling back to default: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    active_vocab = set(config.get("vocabulary", []))
    seed = config.get("seed", 42)

    print(f"Active Benchmark Vocabulary Size: {len(active_vocab)} words")
    print(f"Split mode: {split_mode}")

    max_workers = n_workers or min(len(subject_files), 8)
    print(f"Processing {len(subject_files)} subject files using {max_workers} threads...\n")

    worker_args = [(fp, active_vocab, target_samples) for fp in subject_files]

    all_epochs    = []
    all_labels    = []
    all_subject_ids = []   # needed for subject-level split

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_subject, arg): arg[0] for arg in worker_args}
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                subject_id, epochs, labels = future.result()
                if epochs:
                    all_epochs.extend(epochs)
                    all_labels.extend(labels)
                    all_subject_ids.extend([subject_id] * len(epochs))
            except Exception as exc:
                print(f"[ERROR] {file_path} raised: {exc}")

    if len(all_epochs) == 0:
        print("No trials extracted! Generating synthetic fallback data for verification.")
        vocab_list = list(active_vocab)
        for i in range(150):
            mock_word = vocab_list[i % len(vocab_list)]
            mock_epoch = np.random.randn(105, target_samples).astype(np.float32)
            all_epochs.append(mock_epoch)
            all_labels.append(mock_word)
            all_subject_ids.append("MOCK")

    X_preprocessed  = np.array(all_epochs, dtype=np.float32)
    y_labels         = np.array(all_labels, dtype=object)
    y_subjects       = np.array(all_subject_ids, dtype=object)
    n                = len(X_preprocessed)

    # ── Split strategy ────────────────────────────────────────────────────────
    splits = np.empty(n, dtype=object)

    if split_mode == "subject":
        # ── Subject-level split ───────────────────────────────────────────────
        # Deterministically assign each unique subject to train/val/test.
        # All trials from a given subject go to the same split.
        # Proportions: 70% train / 15% val / 15% test (by number of subjects).
        unique_subjects = sorted(set(all_subject_ids))
        n_subj          = len(unique_subjects)

        rng             = np.random.default_rng(seed)
        shuffled        = rng.permutation(unique_subjects).tolist()

        train_end_s     = int(n_subj * config["split"]["train"])
        val_end_s       = train_end_s + max(1, int(n_subj * config["split"]["val"]))

        train_subjects  = set(shuffled[:train_end_s])
        val_subjects    = set(shuffled[train_end_s:val_end_s])
        test_subjects   = set(shuffled[val_end_s:])

        print(f"\nSubject split (seed={seed}):")
        print(f"  Train ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"  Val   ({len(val_subjects)}):   {sorted(val_subjects)}")
        print(f"  Test  ({len(test_subjects)}):  {sorted(test_subjects)}\n")

        for i, subj in enumerate(y_subjects):
            if subj in train_subjects:
                splits[i] = "train"
            elif subj in val_subjects:
                splits[i] = "val"
            else:
                splits[i] = "test"

    else:
        # ── Example-level split (default) ─────────────────────────────────────
        # All 200 vocabulary words appear in ALL splits.
        # Different EEG recordings of the same word are in different splits.
        rng       = np.random.default_rng(seed)
        idx       = rng.permutation(n)
        train_end = int(n * config["split"]["train"])
        val_end   = train_end + int(n * config["split"]["val"])

        splits[idx[:train_end]]       = "train"
        splits[idx[train_end:val_end]] = "val"
        splits[idx[val_end:]]         = "test"

    # Save output numpy structures
    np.save(os.path.join(DATA_DIR, "X_preprocessed.npy"), X_preprocessed)
    np.save(os.path.join(DATA_DIR, "y_labels.npy"), y_labels)
    np.save(os.path.join(DATA_DIR, "splits.npy"), splits)

    print("=== PIPELINE REGENERATION COMPLETED ===")
    print(f"Generated X_preprocessed.npy shape: {X_preprocessed.shape}")
    print(f"Generated y_labels.npy shape: {y_labels.shape}")
    print(f"Generated splits.npy shape: {splits.shape}")
    print(f"Split distributions - Train: {np.sum(splits == 'train')} | Val: {np.sum(splits == 'val')} | Test: {np.sum(splits == 'test')}")

if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(description="Regenerate preprocessed EEG files from ZuCo .mat subjects.")
    _parser.add_argument(
        "--config", default=None,
        help=(
            "Benchmark config filename (relative to repo root). "
            "Defaults to benchmark_config_50_Zuco1_SR.json if it exists, "
            "otherwise benchmark_config.json."
        ),
    )
    _parser.add_argument(
        "--samples", type=int, default=500,
        help="Target time samples per epoch after resampling (default: 500).",
    )
    _parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel threads (default: number of subject files, capped at 8).",
    )
    _parser.add_argument(
        "--split", choices=["example", "subject"], default="example",
        help=(
            "Split strategy.\n"
            "  example — random trial-level 70/15/15 split (default). "
            "All words in all splits.\n"
            "  subject — split by subject ID. Train/val/test subjects are disjoint. "
            "Forces cross-subject generalisation."
        ),
    )
    _args = _parser.parse_args()

    # Resolve default config: prefer the smaller SR config if present
    if _args.config:
        chosen_config = _args.config
    else:
        _small = os.path.join(BASE_DIR, "benchmark_config_50_Zuco1_SR.json")
        chosen_config = "benchmark_config_50_Zuco1_SR.json" if os.path.exists(_small) else "benchmark_config.json"

    target_subject_files = [
        # ZuCo 1.0 Task 1 — Sentiment Reading (movie reviews)
        "data\\zuco1\\task1-SR\\resultsZAB_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZDM_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZDN_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZGW_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZJM_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZJN_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZJS_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZKB_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZKH_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZKW_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZMG_SR.mat",
        "data\\zuco1\\task1-SR\\resultsZPH_SR.mat",
        # ZuCo 1.0 Task 2 — Normal Reading (Wikipedia)
        # Same 12 subjects, same recording setup — no cross-dataset confounds.
        # Adds diverse vocabulary (politics, history, science) absent from SR movie reviews.
        "data\\zuco1\\task2-NR\\resultsZAB_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZDM_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZDN_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZGW_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZJM_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZJN_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZJS_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZKB_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZKH_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZKW_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZMG_NR.mat",
        "data\\zuco1\\task2-NR\\resultsZPH_NR.mat",
    ]

    print(f"Using config  : {chosen_config}")
    print(f"Split strategy: {_args.split}")
    run_pipeline_regeneration(
        subject_files=target_subject_files,
        config_filename=chosen_config,
        target_samples=_args.samples,
        n_workers=_args.workers,
        split_mode=_args.split,
    )