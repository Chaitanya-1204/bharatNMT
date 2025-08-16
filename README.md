# Anuvvad: A Multilingual Neural Machine Translator

## **Overview**

**Anuvaad** (अनुवाद - Hindi for "Translation") is a powerful, multilingual neural machine translation (NMT) model that translates between **English**, **Hindi**, **Gujarati**, and **Marathi**. This project aims to break down language barriers and facilitate seamless communication across these widely spoken Indian languages.

---

## **Features**

- **Multi-way Translation**: Translate from any of the four supported languages to any other.
- **High Accuracy**: Achieved a **BLEU score of 22.26** and a **BERT score of 0.925**, indicating a high degree of precision and contextual relevance in translations.
- **User-Friendly**: Simple and intuitive interface for easy use.
- **Open Source**: Feel free to contribute and help us improve!

---

## **Supported Languages**

- English (en)
- Hindi (hi)
- Gujarati (gu)
- Marathi (mr)

---

## **Installation**

```bash
conda create -n anuvaad python=3.11 -y
conda activate anuvaad
pip install -r requirements.txt
```

## **Model Training**

```bash
python main.py
```

## **Usage**

After training is complete:

```bash
python app.py
```

Run a local server and open the `index.html` file in your browser.

