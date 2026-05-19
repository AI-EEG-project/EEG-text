# EEG-to-Text Reconstruction

> Inteligenta Artificiala — Echipa de 3 persoane — Mai 2026

## Membrii echipei

| Rol | Nume |
|-----|------|
| Persoana 1 (ZuCo 1.0 — ERP/PSD) | Laslo Tudor Alexandru |
| Persoana 2 (ZuCo 2.0 — Comparatie & Eye-tracking) | Magdas Vlad |
| Persoana 3 (Analiza lingvistica & UI/Backend) | Lupse Ioan Victor |

---

## 1. Prezentare generala

Proiectul construieste un sistem end-to-end care transforma semnale EEG brute (inregistrate in timp ce subiectii citesc cuvinte sau propozitii) intr-o predictie textuala: cuvantul sau propozitia citita. Sistemul foloseste dataset-urile ZuCo si ZuCo 2.0, modele de tip EEGNet sau EEG-Conformer pentru extragerea de features, si aliniere cu spatiul semantic BERT prin contrastive learning.

| Aspect | Detalii |
|--------|---------|
| **Input** | Epoch EEG (.npy) inregistrat in timp ce subiectul citeste un cuvant sau o propozitie |
| **Output** | Top-5 cuvinte candidate cu scoruri de similaritate + propozitie reconstruita |
| **Datasets** | ZuCo 1.0 (P1) + ZuCo 2.0 (P2) + analiza lingvistica combinata (P3) |
| **Modele AI** | EEGNet / EEG-Conformer (features) + retrieval BERT cosine similarity + contrastive learning |
| **Metrici** | Top-1/5 accuracy, cosine similarity EEG-BERT, BLEU score, similaritate semantica sentence-BERT |
| **Interfata** | UI Streamlit + Backend FastAPI (POST /predict -> top-k cuvinte + propozitie reconstruita) |
| **SDG-uri** | SDG 3 (Sanatate) — BCI pentru comunicare nonverbala — SDG 10 (Accesibilitate, dizabilitati) |

---

## 2. Etape si punctaj

| Iteratie | Continut | Punctaj |
|----------|----------|---------|
| **Sapt. 12** | Explorare dataset per persoana — vizualizari, ERP, PSD, analiza lingvistica | **200p** |
| **Sapt. 14 (P1)** | Preprocesare EEG + feature extraction + model retrieval de baza | **300p** |
| **Sapt. 14 (P2)** | Evaluare metrici + imbunatatiri contrastive learning cu BERT | **300p** |
| **Sapt. 14 (P3)** | UI Streamlit + Backend FastAPI pentru demo EEG-to-Text | **inclus** |
| **Video teaser** | 2-3 min (echipele cu >300p): problema, AI, performanta, SDG-uri | **200p** |
| **TOTAL** | | **1000p** |

---

## 3. Saptamana 12 — Explorare datasets ZuCo (200p)

Fiecare persoana exploreaza o perspectiva diferita asupra ecosistemului ZuCo. Livrabil: notebook Jupyter organizat (cod + Markdown), PR #1 pe GitHub.

### Laslo Tudor Alexandru — ZuCo 1.0: ERP, PSD si analiza per tip de cuvant

ZuCo 1.0: 18 subiecti, 105 canale EEG, 500 Hz. Task 1: citire recenzii de filme. Task 2: citire Wikipedia cu intrebari de intelegere. Descarcare OSF.io.

| Sarcina | Pasi detaliati |
|---------|----------------|
| Incarcare si inspectie date | Incarca .mat / .hdf5 cu scipy/h5py. Afiseaza: nr. subiecti, propozitii, cuvinte unice, canale, frecventa esantionare, dimensiunea tensorilor per subject. |
| Vizualizare ERP per tip de cuvant | Calculeaza ERP mediat (media pe trial-uri) pentru 5 electrozi frontali (Fz, F3, F4) si 5 parietali (Pz, P3, P4). Suprapune curbele pentru 4 tipuri de cuvinte: substantive, verbe, adjective, cuvinte functionale. Banda de eroare +/- std. |
| Componente ERP cheie | Identifica si marcheaza pe grafice componentele N200 (~200ms), P300 (~300ms) si N400 (~400ms). Explica semnificatia lor in procesarea lingvistica. Care tip de cuvant genereaza N400 mai pronuntat? |
| Analiza PSD per banda | Calculeaza PSD (metoda Welch) pentru benzile: delta (1-4Hz), theta (4-8Hz), alpha (8-13Hz), beta (13-30Hz), gamma (30-40Hz). Compara puterea per banda intre cele 4 tipuri de cuvinte. Heatmap 4x5 (tipuri x benzi). |
| Topografii EEG | Afiseaza topografia scalp-ului (heatmap 2D canale) pentru media activitatii la 200ms, 300ms si 400ms dupa aparitia cuvantului. Observatii: ce regiuni cerebrale sunt cel mai active? |
| Concluzii + PR #1 | Sectiune Markdown: ce informatii EEG discrimineaza tipurile de cuvinte? Care canale si benzi sunt cele mai informative pentru reconstructia textuala? PR din branch `sapt12-p1`. |

---

### Magdas Vlad — ZuCo 2.0: comparatie taskuri si analiza eye-tracking

ZuCo 2.0: aceiasi 18 subiecti, 2 taskuri noi (Normal Reading si Annotation). Adauga date de eye-tracking sincronizate cu EEG. Descarcare OSF.io.

| Sarcina | Pasi detaliati |
|---------|----------------|
| Incarcare si comparatie structurala | Incarca ZuCo 2.0. Compara cu ZuCo 1.0: nr. propozitii, cuvinte, distributia duratelor fixatiilor. Tabel comparativ ZuCo 1.0 vs. ZuCo 2.0 (canale, subiecti, taskuri, vocabular). |
| ERP comparativ: Task Normal Reading vs. Annotation | Calculeaza ERP pentru ambele taskuri ZuCo 2.0. Suprapune curbele pe aceleasi grafice. Diferentele intre taskuri sugereaza efort cognitiv diferit? |
| Analiza eye-tracking | Scatter plot fixatii pe text (x = pozitie cuvant, y = durata fixatie in ms). Histograma duratelor fixatiilor. Corelatie intre frecventa cuvantului in corpus si durata fixatiei (cuvinte rare -> fixatii mai lungi?). |
| Corelatie EEG — Eye-tracking | Calculeaza corelatie Pearson intre amplitudinea N400 si durata fixatiei per cuvant. Scatter plot EEG amplitude vs. fixation duration. Concluzie: eye-tracking aduce informatie complementara? |
| Variabilitate inter-subiect | Boxplot al amplitudinii N400 per subiect (18 boxuri). Identificarea subiectilor outlier. Discutie: cat de mare este variabilitatea? Impactul asupra generalizarii modelului. |
| Concluzii + PR #1 | Sectiune Markdown: care dataset (ZuCo 1.0 sau 2.0) si care task este mai potrivit pentru antrenarea modelului? Recomandare motivata. PR din branch `sapt12-p2`. |

---

### Lupse Ioan Victor — Analiza lingvistica: vocabular, granularitate si fezabilitate reconstructie

Explorare din perspectiva NLP: ce se poate realist reconstrui din EEG? Analiza vocabularului, granularitatii (cuvant vs. propozitie) si alegerea setului de clase pentru model.

| Sarcina | Pasi detaliati |
|---------|----------------|
| Analiza vocabularului ZuCo | Incarca textele din ZuCo 1.0 + 2.0. Calculeaza: nr. cuvinte unice, distributia frecventelor (curba Zipf), top-200 cuvinte cele mai frecvente. Wordcloud. Concluzie: un vocabular de 50 vs. 200 vs. 500 cuvinte — ce procent din datele de test acopera? |
| Granularitate: cuvant vs. propozitie | Compara doua abordari: (A) clasificare/retrieval la nivel de cuvant — un epoch EEG per cuvant; (B) agregare EEG pe propozitie — medie sau pooling. Vizualizeaza diferentele de semnal intre cele doua granularitati. Recomanda una pentru prototip. |
| Distributia claselor semantice | Grupeaza cuvintele in categorii semantice (POS tagging cu spaCy: NN, VB, JJ, etc.). Bar chart al distributiei. Analiza: clasificarea la nivel semantic (10 categorii) vs. la nivel lexical (vocabular de 200 cuvinte) — care e mai fezabila? |
| Embeddings BERT vizualizate | Genereaza embeddings BERT (bert-base-uncased) pentru top-100 cuvinte din ZuCo. Aplica t-SNE si coloreaza dupa POS tag. Concluzie: spatiul BERT separa bine categoriile? EEG ar putea fi aliniat cu BERT? |
| Propunere benchmark intern | Defineste setul de evaluare: subset de N cuvinte (recomandare: 50-200) din vocabular, split train/val/test fix, seed reproducibil. Salveaza ca fisier JSON pentru utilizare comuna de toti membrii echipei. |
| Concluzii + PR #1 | Sectiune Markdown: vocabular ales, granularitate aleasa, justificare. Fisier `benchmark_config.json` cu parametrii definiti. PR din branch `sapt12-p3`. |

---

## 4. Saptamana 14 — Model, evaluare si interfata (800p)

Echipa foloseste ZuCo 1.0 (Task 1) si configuratia benchmark definita de Lupse Ioan Victor in Sapt. 12. PR #2 obligatoriu de la fiecare.

### Laslo Tudor Alexandru — Preprocesare, EEGNet embeddings si model retrieval EEG-to-Text

| Sarcina | Pasi detaliati |
|---------|----------------|
| Pipeline preprocesare EEG | MNE-Python: filtrare bandpass 0.5-40 Hz, segmentare epoci per cuvant (-0.2s la +0.8s), baseline correction (-200ms la 0ms), rejectie artefacte (>100 microV), z-score normalizare per canal. Export (N, 105, T) ca numpy array. |
| Feature extraction cu EEGNet | Configureaza EEGNet (F1=8, D=2, F2=16, kernLength=64) cu braindecode. Antreneaza pe clasificare POS (substantive vs. verbe vs. adjective). Extrage embeddings din penultimul strat (dim 128). Salveaza modelul si embeddings. |
| Embeddings BERT pentru vocabular target | Genereaza embeddings BERT (bert-base-uncased, CLS token) pentru toate cuvintele din vocabularul ales. Matricea de embeddings: (V, 768) unde V = marimea vocabularului. Salveaza pe disk. |
| Model retrieval prin cosine similarity | Pentru fiecare epoch EEG: calculeaza embedding EEGNet (128 dim) -> proiectie liniara in spatiu BERT (768 dim) -> cosine similarity cu toti vectorii din vocabular -> returneaza top-5 cuvinte. Antreneaza proiectia liniara prin MSE loss intre embedding EEG proiectat si embedding BERT target. |
| Documentare + PR #2 | Notebook cu: diagrama completa a pipeline-ului, cod comentat, exemple de predictii corecte si gresite, vizualizari t-SNE ale embeddings EEG vs. BERT dupa aliniere. PR din branch `sapt14-p1`. |

### Magdas Vlad — Evaluare completa + contrastive learning cu BERT

| Sarcina | Pasi detaliati |
|---------|----------------|
| Metrici de evaluare text | Top-1 accuracy (cuvantul corect pe locul 1), Top-5 accuracy (cuvantul corect in primele 5). Cosine similarity medie intre embedding EEG proiectat si embedding BERT al cuvantului corect. BLEU-1 si BLEU-2 pentru propozitii reconstruite (asamblare cuvinte top-1 per pozitie). Similaritate semantica sentence-BERT. |
| Baseline si comparatii | Baseline aleator: selectie uniforma din vocabular. Baseline frecventa: returneaza mereu cel mai frecvent cuvant. Compara: baseline aleator vs. baseline frecventa vs. model retrieval simplu. Tabele cu toate metricile. |
| Imbunatatire: contrastive learning CLIP-style | Implementeaza pierdere contrastiva (InfoNCE loss) care maximizeaza similaritatea intre embedding EEG si embedding BERT al cuvantului corect si minimizeaza cu celelalte cuvinte din batch (batch size 64). Antreneaza 20 epoci cu AdamW si scheduler cosine. |
| Imbunatatire: EEG-Conformer | Inlocuieste EEGNet cu EEG-Conformer (modul CNN + Transformer). Compara calitativ embeddings: t-SNE EEGNet vs. EEG-Conformer, colorat dupa POS tag. Tabele comparative Top-1/Top-5 pentru ambele arhitecturi. |
| Raport de evaluare + PR #2 | Sectiune Results in notebook: tabele comparative (3 modele x 5 metrici), grafice loss curves, exemple de reconstructie propozitii bune si slabe. Analiza erorilor: ce tipuri de cuvinte sunt mai greu de reconstruit? PR din branch `sapt14-p2`. |

### Lupse Ioan Victor — UI Streamlit + Backend FastAPI pentru demo EEG-to-Text

| Sarcina | Pasi detaliati |
|---------|----------------|
| Backend FastAPI: structura si endpoint /predict | Creeaza `app/backend/main.py`. Incarca la startup modelul EEGNet + proiectia liniara + matricea BERT. Endpoint POST /predict: primeste fisier .npy (epoch EEG shape 105xT), aplica preprocesare + model, returneaza JSON cu: `top_5_words` (lista cuvant + scor), `reconstructed_sentence` (string), `inference_time_ms`. |
| Backend FastAPI: validare si robustete | Validare input cu Pydantic: verifica shape, dtype, valori NaN/Inf. Returneaza HTTP 422 cu mesaje clare la input invalid. Endpoint GET /health (status + versiune model). Endpoint GET /examples (lista cu 5 exemple predefinite din dataset). |
| UI Streamlit: upload si selectie input | Titlu si descriere scurta a proiectului. Doua moduri de input: (1) Upload fisier .npy — drag and drop; (2) Selectie dintre 5 exemple predefinite din ZuCo (cu cuvantul real ascuns initial). Buton **Analizeaza EEG**. |
| UI Streamlit: afisare rezultate | Dupa predictie: (1) Top-5 cuvinte candidate cu bara de progres pentru scor; (2) Propozitia reconstruita afisata prominent; (3) Spectrograma semnalului EEG input (vizualizare canal Pz); (4) Buton **Reveleaza raspunsul corect** (arata cuvantul real din dataset). Timp de inferenta afisat. |
| Instructiuni rulare + PR #2 | `requirements.txt` actualizat (fastapi, uvicorn, streamlit, requests, mne, torch). README cu instructiuni pas cu pas: descarcare model, pornire backend (`uvicorn app.backend.main:app`), pornire UI (`streamlit run app/streamlit_app.py`). Screenshot in README. PR din branch `sapt14-p3`. |

---

## 5. KPI-uri tinta

| Metrica | Tinta minima | Tinta optima |
|---------|-------------|-------------|
| Top-1 Accuracy (cuvant) | > 10% | > 25% |
| Top-5 Accuracy (cuvant) | > 30% | > 55% |
| Cosine Similarity EEG-BERT | > 0.25 | > 0.45 |
| BLEU-1 (propozitii) | > 0.10 | > 0.25 |
| Latenta backend /predict | < 2s | < 500ms |
| Pull Requests per persoana | minim 2 | 3+ |

---

## 6. Stack tehnologic

| Modul | Librarii |
|-------|----------|
| **Procesare EEG** | mne-python, scipy, numpy, h5py |
| **Modele EEG** | braindecode (EEGNet), EEG-Conformer (github: eeyhsong) |
| **NLP / Embeddings** | transformers (BERT), sentence-transformers, spaCy, nltk (BLEU) |
| **ML / DL** | PyTorch, scikit-learn, torchmetrics |
| **Vizualizare** | matplotlib, seaborn, umap-learn, sklearn (t-SNE) |
| **Backend** | FastAPI, uvicorn, pydantic, python-multipart |
| **UI** | Streamlit, requests, Pillow |
| **Versioning** | Git + GitHub Classroom, branch per persoana, minim 2 PR fiecare |

---

## 7. Calendar

| Termen | Livrabile | Responsabil |
|--------|-----------|-------------|
| **Sapt. 12** | 3 notebook-uri explorare ZuCo + `benchmark_config.json` + README | Laslo Tudor Alexandru (ZuCo 1.0 ERP/PSD), Magdas Vlad (ZuCo 2.0 comparatie), Lupse Ioan Victor (analiza lingvistica) |
| **Sapt. 14** | Pipeline ML complet + evaluare + UI + backend functional | Laslo Tudor Alexandru (model retrieval), Magdas Vlad (evaluare+BERT), Lupse Ioan Victor (UI+backend) |
| **2-6 Iun** | Video teaser 2-3 min (echipele cu >300p) | Toti membrii echipei |

---

## 8. Structura repo

```
EEG-proiect/
├── README.md
├── requirements.txt
├── benchmark_config.json          # creat de Lupse Ioan Victor (Sapt. 12)
├── notebooks/
│   ├── sapt12-p1_zuco1_erp_psd.ipynb       # Laslo Tudor Alexandru
│   ├── sapt12-p2_zuco2_eyetracking.ipynb   # Magdas Vlad
│   ├── sapt12-lupse_linguistic_analysis.ipynb # Lupse Ioan Victor
│   └── lupse_conclusions.md       # Concluzii, termeni si recomandari — Lupse Ioan Victor
└── app/
    ├── backend/
    │   └── main.py                # FastAPI — Lupse Ioan Victor
    └── streamlit_app.py           # UI Streamlit — Lupse Ioan Victor
```

> **Lupse Ioan Victor — Concluzii & Ghid termeni:** [`notebooks/lupse_conclusions.md`](notebooks/lupse_conclusions.md)

---

## 9. Note importante

- Fiecare persoana face **minim 2 Pull Request-uri** individuale: PR #1 la Sapt. 12, PR #2 la Sapt. 14.
- Notebook-urile: celule Markdown cu explicatii alternand cu celule de cod, organizate clar pe sectiuni.
- Fisierul `benchmark_config.json` creat de **Lupse Ioan Victor** este folosit de toti: defineste vocabularul, split-ul si seed-ul de reproducibilitate.
- La prezentarea din **Sapt. 12** fiecare prezinta sectiunea sa de explorare (5-7 min / persoana).
- La prezentarea din **Sapt. 14** fiecare prezinta modulul sau: model / evaluare / UI+backend.
- Dataset ZuCo: fisierele .mat sunt mari (~2-5GB/subiect). Lucrati cu un subset de **5 subiecti** pentru prototip.
- Videoul teaser: problema rezolvata, tipul de AI folosit, performanta obtinuta, SDG-urile impactate cu justificare.

---

## Rulare rapida

```bash
# 1. Instaleaza dependintele
pip install -r requirements.txt

# 2. Porneste backend-ul
uvicorn app.backend.main:app --reload --port 8000

# 3. Porneste UI-ul (terminal separat)
streamlit run app/streamlit_app.py
```

Apoi deschide `http://localhost:8501` in browser.
