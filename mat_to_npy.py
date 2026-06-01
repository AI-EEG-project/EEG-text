"""
mat_to_npy.py — Extrage epoci EEG din fisierele .mat ZuCo 1 si le salveaza ca .npy
Lupse Ioan Victor — Sapt. 14

Modificat: Revertit la zero-padding (mode="constant", constant_values=0.0) pentru
a asigura consistenta absoluta cu datele pe care a fost antrenat modelul din Colab.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io

N_CHANNELS    = 105
EPOCH_SAMPLES = 500     # -200ms..+800ms la 500 Hz


def clean_word(raw) -> str:
    """Decodifica un cuvant din format HDF5 (array Unicode sau bytes)."""
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", errors="ignore").strip().lower()
    if isinstance(raw, str):
        return raw.strip().lower()
    if hasattr(raw, "__iter__"):
        try:
            return "".join(chr(int(c)) for c in np.array(raw).flat).strip().lower()
        except Exception:
            pass
    return str(raw).strip().lower()


def fix_punctuation(w: str) -> str:
    return w.strip(".,;:!?\"'()[] \t\n")


def load_word_epochs(mat_path: Path):
    """
    Generator care yield-uieste (word_str, eeg_array) pentru fiecare
    cuvant gasit in fisierul .mat ZuCo 1.

    Suportă dual: formatul v7.3 (via h5py) și formatele mai vechi v5/v7 (via scipy).
    """
    try:
        # Încercăm mai întâi formatul v7.3 (HDF5)
        with h5py.File(mat_path, "r") as f:
            root = f.get("sentenceData")
            if root is None:
                root = f.get("data")
            if root is None:
                keys = list(f.keys())
                if keys:
                    root = f[keys[0]]

            if root is None:
                return

            for sent_ref in np.array(root).flat:
                try:
                    sent = f[sent_ref] if isinstance(sent_ref, h5py.Reference) else sent_ref
                except Exception:
                    continue

                word_group = sent.get("word") if hasattr(sent, "get") else None
                if word_group is None:
                    continue

                for word_ref in np.array(word_group).flat:
                    try:
                        w_obj = f[word_ref] if isinstance(word_ref, h5py.Reference) else word_ref
                    except Exception:
                        continue

                    content = w_obj.get("content") if hasattr(w_obj, "get") else None
                    if content is None:
                        continue
                    try:
                        word = fix_punctuation(clean_word(content[()]))
                    except Exception:
                        continue
                    if not word:
                        continue

                    eeg = None
                    for key in ("rawEEG", "rawData", "meanFixation"):
                        ds = w_obj.get(key) if hasattr(w_obj, "get") else None
                        if ds is None:
                            continue
                        try:
                            arr = np.array(ds, dtype=np.float32)
                            if arr.ndim == 3 and arr.shape[1] == N_CHANNELS:
                                arr = arr.transpose(1, 0, 2).reshape(N_CHANNELS, -1)
                            elif arr.ndim == 3 and arr.shape[2] == N_CHANNELS:
                                arr = arr.reshape(-1, N_CHANNELS).T
                            if arr.ndim != 2:
                                continue
                            if arr.shape[0] != N_CHANNELS:
                                if arr.shape[1] == N_CHANNELS:
                                    arr = arr.T
                                else:
                                    continue
                            if not np.isfinite(arr).all():
                                continue
                            if np.abs(arr).max() > 5000:
                                continue
                            eeg = arr
                            break
                        except Exception:
                            continue

                    if eeg is None:
                        continue

                    T = eeg.shape[1]
                    if T >= EPOCH_SAMPLES:
                        eeg = eeg[:, :EPOCH_SAMPLES]
                    else:
                        # Revertim la padding-ul constant cu 0.0, conform configuratiei de training
                        eeg = np.pad(eeg, ((0, 0), (0, EPOCH_SAMPLES - T)), mode="constant", constant_values=0.0)

                    yield word, eeg

    except OSError:
        # Dacă dă eroarea de semnătură, înseamnă că e formatul vechi MATLAB (v5/v7)
        data = scipy.io.loadmat(
            str(mat_path),
            struct_as_record=False,
            squeeze_me=True
        )

        sentence_data = data.get("sentenceData")
        if sentence_data is None:
            sentence_data = data.get("data")

        if sentence_data is None:
            for k, v in data.items():
                if not k.startswith("__"):
                    sentence_data = v
                    break

        if sentence_data is None:
            return

        if not isinstance(sentence_data, (list, np.ndarray)):
            sentence_data = [sentence_data]

        for sent in sentence_data:
            if not hasattr(sent, "word"):
                continue

            word_group = sent.word
            if not isinstance(word_group, (list, np.ndarray)):
                word_group = [word_group]

            for w_obj in word_group:
                if not hasattr(w_obj, "content"):
                    continue

                try:
                    word = fix_punctuation(clean_word(w_obj.content))
                except Exception:
                    continue
                if not word:
                    continue

                eeg = None
                for key in ("rawEEG", "rawData", "meanFixation"):
                    if not hasattr(w_obj, key):
                        continue
                    try:
                        ds = getattr(w_obj, key)
                        arr = np.array(ds, dtype=np.float32)

                        if arr.shape[0] != N_CHANNELS:
                            if arr.shape[1] == N_CHANNELS:
                                arr = arr.T
                            else:
                                continue
                        if not np.isfinite(arr).all():
                            continue
                        if np.abs(arr).max() > 5000:
                            continue
                        eeg = arr
                        break
                    except Exception:
                        continue

                if eeg is None:
                    continue

                T = eeg.shape[1]
                if T >= EPOCH_SAMPLES:
                    eeg = eeg[:, :EPOCH_SAMPLES]
                else:
                    # Folosim constant zero-padding pentru compatibilitate
                    eeg = np.pad(eeg, ((0, 0), (0, EPOCH_SAMPLES - T)), mode="constant", constant_values=0.0)

                yield word, eeg


def cmd_list(mat_path: Path):
    """Listeaza toate cuvintele unice gasite in fisier."""
    print(f"\nScanez {mat_path.name} ...\n")
    counts: dict[str, int] = {}
    for word, _ in load_word_epochs(mat_path):
        counts[word] = counts.get(word, 0) + 1

    if not counts:
        print("Nu am gasit niciun cuvant cu EEG valid.")
        return

    print(f"{'Cuvant':<20} {'Aparitii':>8}")
    print("-" * 30)
    for w, c in sorted(counts.items()):
        print(f"{w:<20} {c:>8}")
    print(f"\nTotal cuvinte unice: {len(counts)}")


def cmd_extract(mat_path: Path, targets: list[str], out_dir: Path):
    """Extrage prima epoca valida pentru fiecare cuvant din targets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = set(targets)
    found:  dict[str, np.ndarray] = {}

    print(f"\nScanez {mat_path.name} pentru: {sorted(needed)} ...\n")

    for word, eeg in load_word_epochs(mat_path):
        if word in needed and word not in found:
            out_path = out_dir / f"{word}.npy"
            np.save(str(out_path), eeg)
            amp = float(np.abs(eeg).max())
            print(f"  ✓  '{word}'  shape={eeg.shape}  amp_max={amp:.1f}uV  -> {out_path}")
            found[word] = eeg
            if found.keys() == needed:
                break

    missing = needed - found.keys()
    if missing:
        print(f"\n⚠  Nu am gasit: {sorted(missing)}")
        print("   Incearca alt subiect (alt fisier .mat) sau verifica numele cuvantului.")
    else:
        print(f"\n✓  Toate {len(found)} epoci extrase in: {out_dir}/")
        print("   Poti da upload la aceste .npy in aplicatia Streamlit.")


def main():
    parser = argparse.ArgumentParser(
        description="Extrage epoci EEG din .mat ZuCo 1 -> .npy pentru upload in Streamlit"
    )
    parser.add_argument("mat_file", help="Fisierul .mat al unui subiect (ex: results_ZAB_SR.mat)")
    parser.add_argument("--list",   action="store_true", help="Listeaza toate cuvintele din fisier")
    parser.add_argument("--word",   action="append", default=[], metavar="WORD",
                        help="Cuvant de extras (poate fi specificat de mai multe ori)")
    parser.add_argument("--out",    default=".", help="Director output (default: .)")
    args = parser.parse_args()

    mat_path = Path(args.mat_file)
    if not mat_path.exists():
        print(f"EROARE: fisierul '{mat_path}' nu exista.")
        sys.exit(1)

    if args.list:
        cmd_list(mat_path)
    elif args.word:
        cmd_extract(mat_path, [w.lower() for w in args.word], Path(args.out))
    else:
        parser.print_help()
        print("\nExemplu rapid:")
        print(f"  python mat_to_npy.py {mat_path.name} --list")
        print(f"  python mat_to_npy.py {mat_path.name} --word film --word war --out npy_output/")


if __name__ == "__main__":
    main()