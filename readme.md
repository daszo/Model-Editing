## Project Structure

```text
├── GR/
│   ├── get_vectors.py   # Obtain vector representations of corpus or queries
│   ├── get_docid.py     # Retrieve corresponding document IDs based on vectors
│   ├── training.py      # Train generative retrieval model using document IDs
│   ├── get_knowns.py    # Get the knowledge that needs to be edited (known information)
├── DOME.py              # New document model editing
└── data/                # Store training data and auxiliary files
```

---

## Running Steps

### Stage 1 — Train the Generative Retrieval Model

1. **Get Vectors**  
   ```bash
   python GR/get_vectors.py
   ```

2. **Retrieve Document IDs via Vectors**  
   ```bash
   python GR/get_docid.py
   ```

3. **Train Model Based on Document IDs**  
   ```bash
   python GR/training.py
   ```

---

### Stage 2 — Obtain Knowledge to Edit

```bash
python GR/get_knowns.py
```

---

### Stage 3 — New Document Model Editing

```bash
python DOME.py
```

---

## Data Acquisition

This project uses the **NQ (Natural Questions)** dataset as an example.  
You can download it from the following link (Google official open data):

[https://ai.google.com/research/NaturalQuestions/download](https://ai.google.com/research/NaturalQuestions/download)

---
