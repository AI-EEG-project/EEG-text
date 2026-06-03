# Terminologie metrici — EEG-to-Text

---

## Top-1 Accuracy
Procentul de cazuri în care cuvântul corect este prima predicție a modelului.
> Ex: Top-1 = 13% → din 108 teste, modelul a ghicit exact cuvântul corect de 14 ori.

## Top-5 Accuracy
Procentul de cazuri în care cuvântul corect se află printre primele 5 predicții.
> Ex: Top-5 = 30% → cuvântul corect e undeva în top-5, nu neapărat pe primul loc.

## Cosine Similarity
Măsoară cât de „în aceeași direcție" sunt doi vectori în spațiul embedding. Valoare între -1 și 1, unde 1 = identici, 0 = fără legătură, -1 = opuși.
> În contextul nostru: cât de aproape e EEG-ul proiectat de embedding-ul BERT al cuvântului corect.

## BLEU-1 / BLEU-2
Bilingual Evaluation Understudy — scor folosit în NLP pentru a evalua calitatea textului generat față de un text de referință. BLEU-1 compară cuvinte individuale, BLEU-2 perechi de cuvinte consecutive (bigramme). Valoare între 0 și 1.
> Ex: BLEU-1 = 0.15 → 15% din cuvintele prezise apar și în referință.

## Sentence-BERT Similarity
Similitudine cosinus între două propoziții encodate cu Sentence-BERT (un model antrenat special pentru compararea propozițiilor). Mai relevant decât BLEU pentru texte scurte.
> Ex: 0.7+ = propoziții semantic apropiate.

## InfoNCE Loss
Information Noise-Contrastive Estimation — funcție de pierdere contrastivă. Forțează modelul să fie mai similar cu exemplul corect decât cu toate celelalte din batch (sau din vocabular). Valoarea minimă teoretică = 0, valoarea aleatoare = ln(N) unde N = număr de negative.
> Ex: InfoNCE = 5.2 ≈ ln(200) = 5.3 → performanță aleatoare, modelul nu discriminează.

## MSE (Mean Squared Error)
Eroarea pătratică medie — măsoară distanța euclidiană dintre vectorul prezis și cel țintă. Simplu, dar nu forțează discriminarea între cuvinte diferite.
> Problema: modelul poate minimiza MSE convergând spre media tuturor embeddings-urilor BERT.

## Temperature (în InfoNCE)
Parametru care controlează „ascuțimea" distribuției de probabilitate. Temperature mică (ex: 0.07) = distribuție foarte ascuțită = discriminare mai puternică dar gradiente instabile. Temperature mare (ex: 0.5) = distribuție mai moale = antrenare mai stabilă.

## t-SNE
t-Distributed Stochastic Neighbor Embedding — algoritm de reducere a dimensionalității la 2D pentru vizualizare. Păstrează structura locală a datelor (punctele apropiate în spațiul original rămân apropiate în 2D). Nu e folosit pentru antrenare, doar pentru inspecție vizuală.

## EEG Embedding (128 dim)
Vectorul de 128 numere produs de EEGNet din stratul penultim. Conține informația extrasă din semnalul EEG. La noi, conține informație la nivel de POS (tip gramatical), nu per-cuvânt.

## BERT Embedding (768 dim)
Vectorul de 768 numere produs de BERT pentru un cuvânt (CLS token). Conține informație semantică bogată — cuvinte similare ca sens au embeddings apropiate.

## Proiecție liniară (128 → 768)
Un singur strat `nn.Linear` care transformă embedding-ul EEG de 128 dim în spațiul BERT de 768 dim. Fără activare non-liniară — poate doar roti/scala spațiul, nu crea structuri noi.

## KPI (Key Performance Indicator)
Praguri minime de performanță stabilite în README:
- Top-1 > 10% ✅
- Top-5 > 30% ❌ (necesită EEG-Conformer, Magdas P2)
- Cosine > 0.25 ✅

---

*Laslo Tudor Alexandru — Săpt. 14, Branch: Sapt_14_Laslo*
