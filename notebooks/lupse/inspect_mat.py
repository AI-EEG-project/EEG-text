"""
MATLAB File Inspector for ZuCo EEG Data
Author: Lupse Ioan Victor (Person 3)

This script thoroughly inspects any ZuCo .mat file, logs its keys,
checks nested attributes of sentences and word structures, and writes
a full text report to disk so you can see the structure even if the console wraps or buffers.
"""

import os
import sys
import scipy.io as sio
import numpy as np

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAT_PATH = os.path.join(BASE_DIR, "data", "zuco1", "task1-SR", "resultsZAB_SR.mat")
REPORT_PATH = os.path.join(BASE_DIR, "notebooks", "lupse", "mat_structure_report.txt")

# Ensure output directory exists
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def log_and_print(message, file_handle):
    """Prints to console and writes to the report file simultaneously."""
    print(message)
    file_handle.write(message + "\n")


def inspect():
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        log_and_print("==================================================", f)
        log_and_print("          ZUCO MATLAB FILE INSPECTION REPORT       ", f)
        log_and_print("==================================================", f)
        log_and_print(f"Target MAT File Path: {MAT_PATH}", f)

        if not os.path.exists(MAT_PATH):
            log_and_print(f"ERROR: The file does not exist at {MAT_PATH}!", f)
            # Search alternative locations
            log_and_print("Searching for alternative .mat files in data/...", f)
            for root, dirs, files in os.walk(os.path.join(BASE_DIR, "data")):
                for file in files:
                    if file.endswith(".mat"):
                        log_and_print(f" -> Found alternative: {os.path.join(root, file)}", f)
            return

        log_and_print(f"File Size: {os.path.getsize(MAT_PATH) / (1024 * 1024):.2f} MB", f)

        # 1. Load Raw MATLAB structure (no squeezing)
        log_and_print("\n--- 1. LOADING RAW MATLAB KEYS ---", f)
        try:
            raw_mat = sio.loadmat(MAT_PATH)
            keys = [k for k in raw_mat.keys() if not k.startswith("__")]
            log_and_print(f"Success! Raw keys found: {keys}", f)
            for k in keys:
                log_and_print(f"  Key '{k}' type: {type(raw_mat[k])}", f)
        except Exception as e:
            log_and_print(f"CRITICAL ERROR loading raw MAT file: {e}", f)
            return

        # 2. Load Squeezed MATLAB structure
        log_and_print("\n--- 2. LOADING SQUEEZED & OBJECTS ---", f)
        try:
            mat_data = sio.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
            log_and_print("Squeezed structure loaded successfully.", f)
        except Exception as e:
            log_and_print(f"CRITICAL ERROR loading squeezed MAT file: {e}", f)
            return

        # Find sentenceData key case-insensitively
        sentence_key = None
        for key in mat_data.keys():
            if key.lower() == "sentencedata":
                sentence_key = key
                break

        if sentence_key is None:
            log_and_print("ERROR: No sentenceData variable found in keys!", f)
            return

        sentence_data = mat_data[sentence_key]
        log_and_print(f"Found sentenceData key: '{sentence_key}' of type: {type(sentence_data)}", f)

        if isinstance(sentence_data, np.ndarray):
            log_and_print(f"Array shape: {sentence_data.shape}", f)
            log_and_print(f"Array dimensions: {sentence_data.ndim}", f)

            # Flatten or inspect first item
            flat_sentences = sentence_data.ravel()
            log_and_print(f"Total entries in sentenceData: {len(flat_sentences)}", f)
            if len(flat_sentences) == 0:
                log_and_print("ERROR: sentenceData array is empty!", f)
                return
            first_sentence = flat_sentences[0]
        else:
            first_sentence = sentence_data
            log_and_print("sentenceData is a single object (not an array).", f)

        # 3. Inspect the sentence structure
        log_and_print("\n--- 3. INSPECTING FIRST SENTENCE ENTRY ---", f)
        log_and_print(f"First Entry Class: {type(first_sentence)}", f)

        attrs = [a for a in dir(first_sentence) if not a.startswith("_")]
        log_and_print(f"Available attributes on Sentence: {attrs}", f)

        # Check for 'word' or similar
        words = None
        for attr_candidate in ['word', 'words', 'Word', 'Words']:
            if hasattr(first_sentence, attr_candidate):
                words = getattr(first_sentence, attr_candidate)
                log_and_print(f"Found words container attribute: '{attr_candidate}' of type: {type(words)}", f)
                break
            elif isinstance(first_sentence, dict) and attr_candidate in first_sentence:
                words = first_sentence[attr_candidate]
                log_and_print(f"Found words container dict key: '{attr_candidate}' of type: {type(words)}", f)
                break

        if words is None:
            log_and_print("ERROR: No words container attribute found on the sentence object!", f)
            return

        # Unwrap words list
        if isinstance(words, np.ndarray):
            log_and_print(f"Words shape: {words.shape}", f)
            flat_words = words.ravel()
            log_and_print(f"Total words in this sentence: {len(flat_words)}", f)
            if len(flat_words) == 0:
                log_and_print("ERROR: Words array is empty!", f)
                return
            first_word = flat_words[0]
        elif isinstance(words, (list, tuple)):
            log_and_print(f"Total words (list/tuple): {len(words)}", f)
            first_word = words[0]
        else:
            first_word = words
            log_and_print("Words field is a single object.", f)

        # 4. Inspect Word entry fields
        log_and_print("\n--- 4. INSPECTING FIRST WORD OBJECT ---", f)
        log_and_print(f"First Word Type: {type(first_word)}", f)

        word_attrs = [a for a in dir(first_word) if not a.startswith("_")]
        log_and_print(f"Available attributes on Word Object: {word_attrs}", f)

        # Inspect wordstring (the text representation)
        word_string = None
        for attr_candidate in ['wordstring', 'wordString', 'text', 'word']:
            if hasattr(first_word, attr_candidate):
                word_string = getattr(first_word, attr_candidate)
                log_and_print(f"Found word string field: '{attr_candidate}'", f)
                break

        if word_string is not None:
            log_and_print(f"Raw Word value: '{word_string}' (Type: {type(word_string)})", f)
            # Try decoding or resolving if it is wrapped inside array
            if isinstance(word_string, np.ndarray):
                log_and_print(f"  -> Word string is an array! Shape: {word_string.shape}. Flat: {word_string.ravel()}",
                              f)
        else:
            log_and_print("ERROR: No word string field ('wordstring' etc) resolved!", f)

        # Inspect EEG signals
        log_and_print("\n--- 5. CHECKING EEG FIELDS ---", f)
        eeg_found = False
        for field in ['EEG500', 'meanEEG', 'EEG', 'EEG_bandPowers', 'eeg', 'signal']:
            if hasattr(first_word, field):
                sig = getattr(first_word, field)
                log_and_print(f"Found EEG attribute: '{field}' of type: {type(sig)}", f)
                if isinstance(sig, np.ndarray):
                    log_and_print(f"  -> Shape: {sig.shape} (ndim: {sig.ndim})", f)
                eeg_found = True
            elif isinstance(first_word, dict) and field in first_word:
                sig = first_word[field]
                log_and_print(f"Found EEG dict key: '{field}' of type: {type(sig)}", f)
                if isinstance(sig, np.ndarray):
                    log_and_print(f"  -> Shape: {sig.shape} (ndim: {sig.ndim})", f)
                eeg_found = True

        if not eeg_found:
            log_and_print("ERROR: No standard EEG array attributes identified on the word object!", f)

        log_and_print("\n==================================================", f)
        log_and_print("               INSPECTION COMPLETE                ", f)
        log_and_print("==================================================", f)


if __name__ == "__main__":
    inspect()
    print(f"\nReport written successfully! Please open the following file to view details:")
    print(f"  {REPORT_PATH}")