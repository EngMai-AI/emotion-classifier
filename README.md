# 🎭 Emotion Classification with Attention

Multi-class emotion classifier (**joy, sadness, anger, fear, surprise, disgust**) built on the
[GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset.
Four architectures are trained and compared: **LSTM**, **GRU**, **BiLSTM + Attention** (all with
pre-trained GloVe embeddings) and a fine-tuned **DistilBERT** (HuggingFace `Trainer`).
Attention weights are visualized to explain which words drive each prediction, and the best model is
deployed as an interactive **Gradio** app.

- 🤗 **Model:** [mai-data-ai/emotion-distilbert](https://huggingface.co/mai-data-ai/emotion-distilbert)
- 📓 **Notebook:** `notebook.ipynb` (full pipeline: EDA → preprocessing → 4 models → evaluation → heatmaps)
- 🖥️ **App:** `app.py` (Gradio)

---

## 1. Data

GoEmotions (`simplified` config): Reddit comments labeled with 27 emotions + neutral.

**Mapping to 6 Ekman emotions**

| Target | GoEmotions labels |
|---|---|
| anger | anger, annoyance, disapproval |
| disgust | disgust |
| fear | fear, nervousness |
| joy | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring |
| sadness | sadness, disappointment, embarrassment, grief, remorse |
| surprise | surprise, realization, confusion, curiosity |

`neutral` comments are removed. To get a clean **single-label** problem, only comments whose labels all
map to the **same** emotion are kept.

| Split | Original | After filtering |
|---|---|---|
| Train | 43,410 | 28,104 |
| Validation | 5,426 | 3,527 |
| Test | 5,427 | 3,539 |

**Class distribution (train)** — strongly imbalanced:

| joy | anger | surprise | sadness | fear | disgust |
|---|---|---|---|---|---|
| 15,809 (56.3%) | 4,590 (16.3%) | 4,160 (14.8%) | 2,455 (8.7%) | 566 (2.0%) | 524 (1.9%) |

Because of this imbalance, training uses **balanced class weights** and **Macro F1** is the main metric
(accuracy alone is misleading: always predicting *joy* already gives ~56%).

## 2. Preprocessing (RNN models)

- Lowercasing, removal of URLs / mentions / `[NAME]` placeholders, punctuation kept as separate tokens.
- Contractions split the way GloVe expects (`don't → do n't`, `i'm → i 'm`), which raised GloVe token
  coverage to **99.1%**.
- Vocabulary of 20,000 words built **from the train set only**; sequences padded to `max_len = 32`
  (95th percentile of length is 24 words).
- Embedding layer initialized with **GloVe 100d** (`glove-wiki-gigaword-100`), compared with
  `trainable=False` (frozen) vs `trainable=True` (fine-tuned).

DistilBERT uses its own tokenizer on the raw text (`max_length=64`).

## 3. Models

| Model | Notes |
|---|---|
| LSTM | Embedding → SpatialDropout → LSTM(128) → Dropout → Softmax |
| GRU | Same, with GRU(128) |
| BiLSTM + Attention | Embedding → BiLSTM(128, return_sequences) → additive attention (per-token weights + weighted sum) → Softmax |
| DistilBERT | `distilbert-base-uncased`, HuggingFace `Trainer`, weighted cross-entropy, lr 2e-5, batch 32, 4 epochs, best epoch chosen by validation Macro F1 |

## 4. Results (test set, 3,539 comments)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **DistilBERT** | **0.7934** | **0.6962** |
| BiLSTM + Attention (trainable embeddings) | 0.7426 | 0.6526 |
| LSTM (trainable embeddings) | 0.7251 | 0.6359 |
| GRU (trainable embeddings) | 0.7321 | 0.6289 |
| GRU (frozen GloVe) | 0.7098 | 0.6164 |
| BiLSTM + Attention (frozen GloVe) | 0.7056 | 0.6164 |
| LSTM (frozen GloVe) | 0.6917 | 0.5867 |

**DistilBERT per-class results**

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| joy | 0.937 | 0.854 | 0.894 | 1937 |
| sadness | 0.615 | 0.674 | 0.643 | 298 |
| anger | 0.675 | 0.704 | 0.689 | 609 |
| fear | 0.625 | 0.765 | 0.688 | 85 |
| surprise | 0.675 | 0.771 | 0.720 | 533 |
| disgust | 0.490 | 0.610 | 0.543 | 77 |

## DistilBERT confusion matrix
<img width="509" height="470" alt="confusion_matrix_distilbert" src="https://github.com/user-attachments/assets/fcae4625-37c6-45c2-8f19-6bb33a49db5e" />


### Discussion

- **DistilBERT is the best model** (+4.4 Macro F1 points over the best recurrent model). Pre-training
  gives contextual representations, while GloVe assigns one fixed vector per word.
- **Fine-tuning the embeddings helps a lot** for the recurrent models (e.g. LSTM Macro F1 0.587 → 0.636).
- Among the models trained from scratch, **BiLSTM + Attention is the best**, and it is also the one that
  can be inspected visually (next section).
- Class weights trade precision for recall on rare classes (e.g. disgust recall 0.61 vs precision 0.49).
  **Disgust** remains the hardest class (only 524 training examples).
- All numbers come from a **single run / single seed**; small differences (e.g. LSTM vs GRU) should not
  be over-interpreted.

## 5. Attention interpretability

Attention weights of the BiLSTM + Attention model show which tokens drive each prediction:

| Sentence | Prediction | Most attended token |
|---|---|---|
| I am so happy and **grateful** for everything you did for me! | joy (95%) | `grateful` (0.76) |
| I'm **terrified**, something is following me and I can't breathe | fear (100%) | `terrified` (0.94) |
| This is **disgusting**, I can't believe anyone would eat that | disgust (100%) | `disgusting` (0.95) |
| I **miss** her so much, I feel so empty and alone | sadness (99%) | `miss` (0.71) |
| **Wow**, I did not expect that at all! What a **surprise**! | surprise (96%) | `wow` (0.46), `surprise` (0.37) |

## Attention heatmap
<img width="1037" height="1048" alt="attention_heatmaps" src="https://github.com/user-attachments/assets/1c162eee-463d-41b0-b10a-220ff53a9ce5" />


Words with the highest average attention per class (on correctly classified test comments):

- **joy:** thank, thanks, cheers, hilarious, beautiful, fantastic
- **sadness:** upset, disappointing, regret, sad, miss, sorry
- **anger:** annoying, angry, frustrating, stupid, hate
- **fear:** scary, fear, afraid, worried, terrified
- **surprise:** realized, surprised, wondering, curious, confused
- **disgust:** awful, disgusting, worse, weird

The model mostly relies on explicit emotion keywords. Note that attention is a useful indication of
important words, not a complete causal explanation.

## 6. Run the app locally

```bash
git clone https://github.com/<your-username>/emotion-classifier.git
cd emotion-classifier

python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      |  Linux/macOS: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python app.py
```

Then open <http://127.0.0.1:7860>. The model weights (~268 MB) are downloaded automatically from the
HuggingFace Hub on first run. Type a sentence to get the dominant emotion and a bar chart of the
confidence for all 6 classes.

## App screenshot
<img width="907" height="901" alt="image" src="https://github.com/user-attachments/assets/8b01847a-467e-430f-829a-2184c7d2ad5b" />
<img width="910" height="882" alt="image" src="https://github.com/user-attachments/assets/98983603-ba8f-421a-9dc0-af38f6cdebf8" />




## 7. Repository structure

```
emotion-classifier/
├── app.py               # Gradio app (loads the model from the HF Hub)
├── requirements.txt
├── notebook.ipynb       # full experiment pipeline
├── images/              # confusion matrices, attention heatmaps, app screenshot
└── README.md
```

## 8. Limitations

- Labels come from Reddit comments and are noisy and overlapping; the 6-class mapping merges many
  fine-grained emotions (e.g. *realization* and *curiosity* are counted as *surprise*).
- Multi-label comments with conflicting emotions were dropped, so the model is not evaluated on them.
- Sarcasm and emotions expressed without explicit keywords are hard for all models.
- Results are from a single training run per model.

## Tech stack

TensorFlow/Keras (LSTM, GRU, BiLSTM + Attention), PyTorch + HuggingFace Transformers (DistilBERT),
scikit-learn, GloVe (gensim), Gradio, matplotlib / seaborn.

## Author

*Eng /MAi Reda*
