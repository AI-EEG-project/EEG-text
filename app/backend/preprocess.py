"""
preprocess.py — Preprocesare semnal EEG adaptata dupa modelul de antrenare Laslo (P1)
Lupse Ioan Victor — Sapt. 14

Corectat matematic pentru a elimina mismatch-urile cu Notebook-ul de antrenare:
    1. Reconstituie segmentul activ original T prin detectarea inceputului flatline-ului.
    2. Aplica filtru zero-phase (filtfilt) pe axa temporala, evitand intarzierile de faza ale sosfilt.
    3. Calculeaza Z-Score strict pe esantioanele active (lungime T), eliminand contaminarea din padding.
    4. Aplica zero-padding (constant 0.0) la final pana la 500 esantioane.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def detect_active_length(eeg: np.ndarray, tolerance: float = 1e-7) -> int:
    """
    Detecteaza lungimea originala T a semnalului activ inainte de aplicarea padding-ului.
    Cauta de la dreapta la stanga unde semnalul inceteaza sa fie o linie constanta (flatline).
    """
    n_ch, n_t = eeg.shape
    if n_t < 500:
        return n_t

    # Calculam diferenta absoluta intre esantioane consecutive
    diffs = np.diff(eeg, axis=1)  # shape (105, 499)

    # Identificam esantioanele unde semnalul este constant (dif ~ 0) pe absolut toate cele 105 canale
    is_constant = np.all(np.abs(diffs) < tolerance, axis=0)  # boolean vector de lungime 499

    # Parcurgem de la final spre inceput pentru a gasi ultima modificare reala a semnalului
    for i in range(len(is_constant) - 1, -1, -1):
        if not is_constant[i]:
            # +2 deoarece np.diff reduce dimensiunea cu 1 si dorim indexul esantionului real de dupa diferenta
            return i + 2

    return n_t


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Aplica filtrul Butterworth bandpass folosind filtfilt (zero-phase forward-backward filter).
    Filtrarea se realizeaza pe ultima axa (axa temporala).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    # filtfilt asigura defazaj zero, eliminand intarzierile de timp introduse de filtrele IIR cauzale
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def preprocess(eeg: np.ndarray, fs: int = 500) -> np.ndarray:
    """
    Pipeline-ul complet de preprocesare aliniat 100% cu cel din faza de antrenare:
    Filtrare -> Z-score pe segmentul activ -> Zero padding la 500.
    """
    if eeg.ndim != 2:
        raise ValueError(f"EEG trebuie sa fie 2D (canale, timp), primit {eeg.ndim}D.")

    # Pasul 1: Detectam lungimea activa T pentru a elimina contaminarea cauzata de padding
    T = detect_active_length(eeg)

    # Extragem doar segmentul activ real
    active_segment = eeg[:, :T]

    # Pasul 2: Filtram segmentul activ folosind filtfilt (zero-phase)
    # Replicam exact parametrii de antrenare ai lui Laslo: 0.5 - 40 Hz, ordinul 4
    filtered_segment = butter_bandpass_filter(active_segment, lowcut=0.5, highcut=40.0, fs=fs, order=4)

    # Pasul 3: Calculam Z-score pe canale strict pentru portiunea activa filtrata
    mean = filtered_segment.mean(axis=1, keepdims=True)
    std = filtered_segment.std(axis=1, keepdims=True) + 1e-8
    normalized_segment = (filtered_segment - mean) / std

    # Pasul 4: Re-aplicam zero-padding strict la coada semnalului pana la 500 de esantioane
    epoch_samples = 500
    if T >= epoch_samples:
        final_eeg = normalized_segment[:, :epoch_samples]
    else:
        # Puffer-ul nefolosit devine din nou 0.0 perfect, exact asa cum a vazut modelul la antrenare
        final_eeg = np.pad(normalized_segment, ((0, 0), (0, epoch_samples - T)), mode="constant", constant_values=0.0)

    return final_eeg.astype(np.float32)
