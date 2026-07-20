# Autonext
Official repo of the work "Autonext: Optimised Recurrent and Context-Aware Statistical Models for Predictive Text Input in Low-Resource Indic Languages".

AutoNext is an open-source implementation of multilingual predictive text entry models for **character prediction** and **word prediction**. The repository contains the implementation of **Optimised Recurrent Language Models (ORLMs)** based on LSTM, BiLSTM, and GRU architectures, along with the proposed **Context-Aware Smoothed Entropy-based Prediction by Partial Matching (CASE-based PPM)** models.

The repository accompanies our research on multilingual predictive text entry systems and includes the source code, datasets, trained models, and evaluation scripts.

## Features

- Character prediction using:
  - LSTM
  - BiLSTM
  - GRU
  - CASE-based PPM-3
  - CASE-based PPM-4
  - CASE-based PPM-5
  - CASE-based PPM-6

- Word prediction using:
  - LSTM
  - BiLSTM
  - GRU

- Multilingual support

  - English
  - Hindi
  - Gujarati
  - Kannada
  - Marathi
  - Punjabi

- Hyperparameter optimisation using Random Search

- Evaluation using

  - Accuracy
  - Validation Loss
  - Perplexity
  - Total Percentage of Correct Characters (TPOC)
  - Keystroke Savings (KSS)
  - Hit Rate


# Repository Structure

```text
AutoNext/
│
├── CharacterPrediction/
│   ├── LSTM/
│   ├── BiLSTM/
│   ├── GRU/
│   └── CASEBasedPPM/
│
├── WordPrediction/
│   ├── LSTM/
│   ├── BiLSTM/
│   └── GRU/
│
├── Dataset/
│   ├── English/
│   ├── Hindi/
│   ├── Gujarati/
│   ├── Kannada/
│   ├── Marathi/
│   └── Punjabi/
│
├── Results/
│
├── requirements.txt
│
└── README.md
```


# Datasets

The repository includes multilingual datasets used for training and evaluation.

| Language | Source |
|-----------|--------|
| English | The Adventures of Sherlock Holmes |
| Hindi | CC100 |
| Gujarati | CC100 |
| Kannada | CC100 |
| Marathi | CC100 |
| Punjabi | CC100 |

Each dataset is divided into training and validation sets using an 80:20 split.


# Character Prediction

The repository contains implementations of

- Optimised LSTM
- Optimised BiLSTM
- Optimised GRU
- CASE-based PPM

The CASE-based PPM incorporates

- Laplace smoothing
- Entropy-based interpolation
- Recursive fallback mechanism

to improve prediction robustness for sparse and unseen contexts.

# Word Prediction

The repository also contains multilingual next-word prediction models implemented using

- LSTM
- BiLSTM
- GRU

The models are trained using optimised hyperparameters obtained through random search.

# Installation

Clone the repository

```bash
git clone https://github.com/HAIx-Lab/Autonext.git
```

Move into the repository

```bash
cd Autonext
```

---

# Running Character Prediction

Example

```bash
python CharEnglishPrediction.py
```

or

```bash
python CharEnglishPrediction.py
```

depending on the respective model directory.

---

# Running Word Prediction

Navigate to the desired model directory and execute the required file. Ex:

```bash
python WordEnglishPrediction.py
```

or

```bash
python WordMarathiPrediction.py
```


# Hyperparameter Optimisation

The Optimised Recurrent Language Models (ORLMs) were obtained using Random Search over three hyperparameters:

- Learning Rate
- Batch Size
- Number of Layers

The best configuration was selected using validation accuracy, validation loss, and validation perplexity.

# Evaluation Metrics

The models are evaluated using

- Validation Accuracy
- Validation Loss
- Validation Perplexity
- Total Percentage of Correct Characters (TPOC)
- Keystroke Savings (KSS)
- Hit Rate

# Applications

The proposed models can be integrated into

- Virtual keyboards
- Assistive communication systems
- Eye-gaze typing interfaces
- AAC devices
- Mobile text entry systems
- Predictive typing applications
- Speech-to-text systems
- Live transcription
- Real-time translation systems
- Embedded and edge AI devices

# Citation

If you use this repository in your research, please cite our work.

```bibtex
@article{AutoNext2026,
  title     = {AutoNext: Multilingual Character and Word Prediction using Optimised Recurrent Language Models and Context-Aware Statistical Models},
  author    = {},
  journal   = {},
  year      = {2026}
}
```

---

# Contact

For questions or collaborations, please contact the HAIx Lab.



