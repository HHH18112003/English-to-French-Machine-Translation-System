# English-to-French Machine Translation System (Custom Architecture)

An end-to-end Neural Machine Translation (NMT) system built from scratch using PyTorch to translate text from English to French. 

⚠️ **Special Highlight:** To meet strict academic constraints and avoid library dependency compatibility errors, this project **fully reimplements a custom vocabulary and text tokenization pipeline from scratch**, completely independent of `torchtext` build tools.

## 📌 Key Features
- **Custom Vocab & Tokenizer Implementation:** Built a dedicated `Vocab` class from the ground up utilizing `collections.Counter` and raw text iterators to handle mapping, `<unk>`, `<pad>`, `<sos>`, and `<eos>` tokens dynamically.
- **Custom Data Pipeline:** Implemented a robust PyTorch `Dataset` paired with a custom `collate_fn` to perform dynamic padding (`pad_sequence`) and sequence packing (`pack_padded_sequence`) for optimizing recurrent execution.
- **Deep Sequence-to-Sequence Architecture:** Designed an **Encoder-Decoder Recurrent Neural Network** using stacked multi-layer LSTMs with integrated Dropout layers to manage feature normalization.
- **Advanced Training Control:** Configured with dynamic **Teacher Forcing** ratios, Adam optimizer, Cross-Entropy loss ignoring padding tokens, Gradient Clipping (`clip_grad_norm_`), and an automated **Early Stopping (Patience)** mechanism.
- **Automated BLEU Evaluation:** Complete pipeline for calculating corpus BLEU score against test datasets via NLTK to provide verified benchmark reports.

## 🛠️ Tech Stack & Libraries
- **Framework:** PyTorch (Core Neural Network architecture)
- **NLP Utilities:** SpaCy (Language specific tokenization models `en_core_web_sm` / `fr_core_news_sm`), NLTK (BLEU metrics)
- **Data Engineering:** NumPy, Python Built-ins (`gzip`, `collections.Counter`)
- **Visualization:** Matplotlib

## 📊 Dataset & Performance Results
- **Dataset:** Multi30k Task 1 Raw Dataset (29,000 Training pairs, 1,014 Validation pairs, 1,000 Test pairs).
- **Vocab Size:** English: 6,191 words | French: 6,555 words.
- **Training Convergence:** The model successfully applied Early Stopping before maximum epochs as validation loss stabilized.
- **Test Evaluation:** Achieved a **Corpus BLEU Score of 32.93**.

### Sample Translations:
- **Source (En):** `A mother and her young song enjoying a beautiful day outside.`
- **Target (Fr):** `Une mère et son fils se amusant à une journée ensoleillée .`
- **Source (En):** `People walking down sidewalk next to a line of stores.`
- **Target (Fr):** `Des gens marchant sur le trottoir à côté d' un stand de restauration .`

   ## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/HHH18112003/English-to-French-Machine-Translation-System.git](https://github.com/HHH18112003/English-to-French-Machine-Translation-System.git)
   cd English-to-French-Machine-Translation-System
2. Install Dependencies & Language Packs:
   pip install torch spacy nltk matplotlib numpy
   python -m spacy download en_core_web_sm
   python -m spacy download fr_core_news_sm

3. Train & Evaluate:
Run the notebook cells or main script. The script will automatically clone the raw multi30k-dataset, construct the custom vocabulary dictionary, initiate model training with custom hyper-parameters, save the best weights to best_model.pth, and prompt the test evaluation metrics.
