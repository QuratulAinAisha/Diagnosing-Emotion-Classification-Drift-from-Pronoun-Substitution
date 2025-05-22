# Pronoun Bias in Emotion Classification

This repository contains code to evaluate **pronoun-induced emotion prediction drift** in pretrained NLP models. It shows how changing the subject pronoun in a sentence ("he", "she", or "they")—while keeping the sentence otherwise identical—can lead to different predicted emotions by widely used classifiers.

---

## 📊 What This Code Does

- Uses a CSV file of **1,000 sentence triplets**, each differing only in subject pronoun.
- Runs predictions using a Hugging Face model (e.g., `distilbert-base-uncased-go-emotions-student`).
- Measures **label mismatches** between:
  - `he` vs `she`
  - `he` vs `they`
  - `she` vs `they`
- Visualizes emotion drift using confusion matrix heatmaps.
- Prints example mismatches for qualitative understanding.

---

## 📁 Dataset Format

The dataset (`gender_emotion_sentences_1000.csv`) should have the following structure:

| he                            | she                           | they                          |
|------------------------------|-------------------------------|-------------------------------|
| He stayed silent.            | She stayed silent.            | They stayed silent.           |
| He helped a friend.          | She helped a friend.          | They helped a friend.         |
| ...                          | ...                           | ...                           |

---

## 🚀 How to Run

1. **Install dependencies:**

```bash
pip install transformers pandas matplotlib seaborn
