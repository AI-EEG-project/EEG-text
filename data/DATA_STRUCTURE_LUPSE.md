# Structura datelor ZuCo

> Referinte: [ZuCo 1.0 — Scientific Data 2018](https://www.nature.com/articles/sdata2018291) | [ZuCo 2.0 — OSF](https://osf.io/2urht/)

---

## ZuCo 1.0 — 12 subiecti, 3 task-uri

| Task | Continut | Propozitii | Fisier text | Fisier adnotari |
|------|----------|------------|-------------|-----------------|
| Task 1 — SR (Sentiment Reading) | Recenzii de film | 600 | `sentencesSR.mat` | `sentiment_normal_reading.csv` |
| Task 2 — NR (Normal Reading) | Articole Wikipedia | 595 | `sentencesNR.mat` | `relations_normal_reading.csv` |
| Task 3 — TSR (Task-Specific Reading) | Wikipedia (relatii semantice) | 407 | `sentencesTSR.mat` | `relations_task_specific.csv` |

**Total ZuCo 1.0: 1602 propozitii unice**

Format `.mat`: camp `sentences`, shape `(N, 1)`, citit cu `scipy.io.loadmat`.

---

## ZuCo 2.0 — 18 subiecti, 2 task-uri

| Task | Continut | Propozitii | Fisiere text |
|------|----------|------------|--------------|
| Task 1 — NR (Normal Reading) | Articole Wikipedia | 370 | `task_materials/nr_1.csv` … `nr_7.csv` |
| Task 2 — TSR (Task-Specific Reading) | Wikipedia (relatii semantice) | 411 | `task_materials/tsr_1.csv` … `tsr_7.csv` |

**Total ZuCo 2.0: 781 propozitii**

Format CSV: separator `;`, coloane `ID ; paragraph_id ; sentence ; relation_type`

> !!!! **Overlap ~100 propozitii** cu ZuCo 1.0 in blocurile `nr_1`, `nr_7`, `tsr_1`, `tsr_7` — reinregistrate pentru analize cross-dataset.

---

## Ce folosim pentru analiza lingvistica (P3)

| Sursa | Fisiere | Propozitii |
|-------|---------|------------|
| ZuCo 1.0 | `sentencesSR.mat`, `sentencesNR.mat`, `sentencesTSR.mat` | 1602 |
| ZuCo 2.0 | `nr_1.csv` … `nr_7.csv`, `tsr_1.csv` … `tsr_7.csv` | 781 |
| **TOTAL (dupa deduplicare)** | | **~2283** |

---

## Cum se incarca

```python
import scipy.io as sio, csv

# ZuCo 1.0 — fisiere .mat
def load_mat_sentences(path):
    mat = sio.loadmat(path)
    return [str(mat['sentences'][i][0][0]).strip()
            for i in range(mat['sentences'].shape[0])]

sr  = load_mat_sentences('data/zuco1/task1-SR/preprocessed/sentencesSR.mat')   # 600
nr  = load_mat_sentences('data/zuco1/task2-NR/preprocessed/sentencesNR.mat')   # 595
tsr = load_mat_sentences('data/zuco1/task3-TSR/preprocessed/sentencesTSR.mat') # 407

# ZuCo 2.0 — fisiere CSV (separator ;)
def load_csv_sentences(paths):
    sentences = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for row in csv.reader(f):
                parts = ''.join(row).split(';')
                if len(parts) >= 3:
                    sentences.append(parts[2].strip().strip('"'))
    return sentences

nr2  = load_csv_sentences([f'data/zuco2/task_materials/nr_{i}.csv'  for i in range(1, 8)])
tsr2 = load_csv_sentences([f'data/zuco2/task_materials/tsr_{i}.csv' for i in range(1, 8)])

all_sentences = sr + nr + tsr + nr2 + tsr2  # ~2383 (cu overlap)
```
