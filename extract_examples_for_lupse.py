"""
extract_examples_for_lupse.py
─────────────────────────────
Laslo ruleaza acest script in mediul lui (unde are datele ZuCo complete).
Produce 5 fisiere .npy cu epoci EEG reale, gata de pus in data/examples/.

Cerinte: h5py, scipy, numpy  (deja instalate)
Rulare:
    python extract_examples_for_lupse.py --zuco_dir /path/to/zuco1/task1-SR

Fisierele produse (trimite-le lui Lupse):
    example_0_film.npy
    example_1_kennedy.npy
    example_2_law.npy
    example_3_school.npy
    example_4_war.npy
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

# ── Cuvintele tinta ──────────────────────────────────────────────────────────
TARGETS = {
    0: "film",
    1: "kennedy",
    2: "law",
    3: "school",
    4: "war",
}

# ── Parametri ZuCo ───────────────────────────────────────────────────────────
N_CHANNELS   = 105
EPOCH_SAMPLES = 500        # -200ms .. +800ms la 500 Hz
FS           = 500

def normalize_word(w):
    return w.strip().lower().rstrip(".,;:!?\"'()[]")


def extract_from_h5(mat_path: Path, target_word: str) -> np.ndarray | None:
    """
    Extrage prima epoca cu cuvantul target din fisierul .mat HDF5 al unui subiect ZuCo.
    Returneaza array (105, 500) float32 sau None daca nu gaseste.
    """
    with h5py.File(mat_path, "r") as f:
        # ZuCo 1 structure: f['data'] sau f['sentenceData']
        root_key = "sentenceData" if "sentenceData" in f else list(f.keys())[0]
        sent_data = f[root_key]

        for sent_ref in sent_data.flat:
            try:
                sent = f[sent_ref]
                word_data = sent["word"]

                for word_ref in word_data.flat:
                    try:
                        w_obj  = f[word_ref]
                        raw_w  = w_obj["content"][()]
                        # Decodifica cuvantul
                        if isinstance(raw_w, bytes):
                            word = raw_w.decode("utf-8")
                        elif raw_w.dtype.kind in ("U", "S", "O"):
                            word = str(raw_w.flat[0])
                        else:
                            # Array de coduri Unicode
                            word = "".join(chr(int(c)) for c in raw_w.flat)

                        if normalize_word(word) != target_word:
                            continue

                        # Citeste rawEEG sau meanFixation
                        for eeg_key in ("rawEEG", "meanFixation", "rawData"):
                            if eeg_key in w_obj:
                                eeg = np.array(w_obj[eeg_key], dtype=np.float32)
                                # Shape trebuie (n_ch, T); transpu daca e invers
                                if eeg.ndim == 2:
                                    if eeg.shape[0] != N_CHANNELS and eeg.shape[1] == N_CHANNELS:
                                        eeg = eeg.T
                                    if eeg.shape[0] == N_CHANNELS:
                                        # Crop / pad la EPOCH_SAMPLES
                                        T = eeg.shape[1]
                                        if T >= EPOCH_SAMPLES:
                                            eeg = eeg[:, :EPOCH_SAMPLES]
                                        else:
                                            eeg = np.pad(eeg, ((0,0),(0, EPOCH_SAMPLES - T)))
                                        # Verifica amplitudine (valori brute in uV)
                                        if np.abs(eeg).max() < 2000 and np.isfinite(eeg).all():
                                            return eeg
                    except Exception:
                        continue
            except Exception:
                continue
    return None


def extract_from_scipy(mat_path: Path, target_word: str) -> np.ndarray | None:
    """Fallback pentru formate .mat mai vechi (scipy.io)."""
    try:
        mat = sio.loadmat(str(mat_path), squeeze_me=True)
        sent_data = mat.get("sentenceData", mat.get("data", None))
        if sent_data is None:
            return None

        for sent in np.array(sent_data).flat:
            if not hasattr(sent, "dtype") or sent.dtype.names is None:
                continue
            if "word" not in sent.dtype.names:
                continue

            for w_obj in np.array(sent["word"]).flat:
                if not hasattr(w_obj, "dtype") or w_obj.dtype.names is None:
                    continue
                if "content" not in w_obj.dtype.names:
                    continue
                try:
                    raw_w = w_obj["content"]
                    word  = str(raw_w).strip().lower().rstrip(".,;:!?\"'()[]")
                    if word != target_word:
                        continue

                    for eeg_key in ("rawEEG", "meanFixation", "rawData"):
                        if eeg_key in w_obj.dtype.names:
                            eeg = np.array(w_obj[eeg_key], dtype=np.float32)
                            if eeg.ndim == 2 and eeg.shape[0] == N_CHANNELS:
                                T = eeg.shape[1]
                                if T >= EPOCH_SAMPLES:
                                    eeg = eeg[:, :EPOCH_SAMPLES]
                                else:
                                    eeg = np.pad(eeg, ((0,0),(0, EPOCH_SAMPLES - T)))
                                if np.abs(eeg).max() < 2000 and np.isfinite(eeg).all():
                                    return eeg
                except Exception:
                    continue
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zuco_dir",
        default="data/zuco1/task1-SR",
        help="Director ZuCo task (contine preprocessed/ cu fisierele results_Z*.mat)",
    )
    parser.add_argument(
        "--out_dir",
        default="data/examples",
        help="Director unde se salveaza .npy-urile",
    )
    args = parser.parse_args()

    zuco_dir = Path(args.zuco_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gaseste fisierele subject
    mat_files = sorted(
        list(zuco_dir.rglob("results_Z*.mat")) +
        list(zuco_dir.rglob("ZAB*.mat")) +
        list(zuco_dir.rglob("Z*.mat"))
    )
    if not mat_files:
        print(f"EROARE: Nu am gasit fisiere .mat in {zuco_dir}")
        print("Cauta fisiere cu prefix 'results_Z' sau 'Z' (per-subiect).")
        return

    print(f"Gasit {len(mat_files)} fisiere subiect: {[f.name for f in mat_files]}")

    found = {}

    for mat_path in mat_files:
        if len(found) == len(TARGETS):
            break
        print(f"\nScanez {mat_path.name} ...")

        for idx, word in TARGETS.items():
            if idx in found:
                continue

            # Incearca h5 mai intai, apoi scipy
            eeg = extract_from_h5(mat_path, word)
            if eeg is None:
                eeg = extract_from_scipy(mat_path, word)

            if eeg is not None:
                out_path = out_dir / f"example_{idx}_{word}.npy"
                np.save(str(out_path), eeg)
                print(f"  ✓ '{word}' -> {out_path.name}  shape={eeg.shape}  "
                      f"amp=[{eeg.min():.1f}, {eeg.max():.1f}] uV")
                found[idx] = word

    print(f"\nRezultat: {len(found)}/5 epoci extrase.")
    if found:
        print("Cuvinte gasite:", found)

    # Actualizeaza manifest.json
    manifest_path = out_dir / "manifest.json"
    descriptions = {
        "film":    "Subiect citeste un text despre industria cinematografica",
        "kennedy": "Subiect citeste despre un personaj politic american",
        "law":     "Subiect citeste despre sistemul juridic",
        "school":  "Subiect citeste despre institutii educationale",
        "war":     "Subiect citeste despre un conflict armat",
    }

    examples_list = []
    for idx, word in sorted(found.items()):
        npy_name = f"example_{idx}_{word}.npy"
        examples_list.append({
            "id":          idx,
            "true_word":   word,
            "description": descriptions.get(word, word),
            "filename":    npy_name,
            "shape":       [N_CHANNELS, EPOCH_SAMPLES],
            "source":      "ZuCo 1.0 — real EEG epoch",
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"examples": examples_list}, f, indent=2, ensure_ascii=False)

    print(f"\nManifest actualizat: {manifest_path}")
    print("Trimite fisierele .npy si manifest.json lui Lupse -> data/examples/")


if __name__ == "__main__":
    main()
