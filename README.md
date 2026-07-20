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
│   ├── CharEnglishPrediction.py
│   ├── CharHindiPrediction.py
│   ├── CharGujaratiPrediction.py
│   ├── CharKannadaPrediction.py
│   ├── CharMarathiPrediction.py
│   └── CharPunjabiPrediction.py
│
├── WordPrediction/
│   ├── WordEnglishPrediction.py
│   ├── WordHindiPrediction.py
│   ├── WordGujaratiPrediction.py
│   ├── WordKannadaPrediction.py
│   ├── WordMarathiPrediction.py
│   └── WordPunjabiPrediction.py
│
├── data/
│   ├── english.txt
│   ├── hindi.txt
│   ├── gujarati.txt
│   ├── kannada.txt
│   ├── marathi.txt
│   └── punjabi.txt
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
@article{10.1145/3822405,
author = {Somkuwar, Tanmay and Meena, Yogesh},
title = {Autonext: Optimised Recurrent and Context-Aware Statistical Models for Predictive Text Input in Low-Resource Indic Languages},
year = {2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2375-4699},
url = {https://doi.org/10.1145/3822405},
doi = {10.1145/3822405},
abstract = {Text entry systems allow users to input information and interact with digital environments, with predictive input technologies further enhancing efficiency by anticipating user intent and offering real-time suggestions for words or letters. To date, the use of predictive approaches for letters and words in low-resource languages is still limited. This work proposes a set of deep learning-based optimised recurrent language models (ORLMs), along with statistical context-aware smoothed entropy-based prediction by partial matching methods (CASE-based PPM) for next-letter and next-word prediction. We evaluated these RNN-based deep learning models and PPM-based statistical methods in real time using various input sentences in six languages: English, Hindi, Marathi, Gujarati, Punjabi, and Kannada. We conducted four experiments to assess these approaches across multiple standard and practical performance metrics, with a comparative analysis aimed at real-world deployment in predictive applications. The total percentage of correct characters or letters achieved for CASE-based PPM-5 and PPM-6 was 90.63 ± 3.33\%, 92.85 ± 4.93\%, respectively, across all languages. The optimised deep learning models, using long short-term memory and gated recurrent unit techniques, showed better suitability for word prediction applications, achieving accuracies of 93.63 ± 2.15\% and 94.01 ± 2.08\%, keystroke savings of 51.59 ± 16.69\% and 50.25 ± 14.95\%, and hit rates of 62.75 ± 27.41\% and 60.575 ± 25.38\%, respectively, across all six languages. These results demonstrate the effectiveness of the proposed approaches, which consistently outperform state-of-the-art predictive methods across various performance metrics and hold significant potential for developing text entry applications in low-resource languages.},
note = {Just Accepted},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = jul,
keywords = {Recurrent Neural Networks, Prediction by Partial Matching, Next Word Prediction, Next Letter Prediction, Text Input, Text Entry Interfaces, Low-Resource Indic Languages}
}
```

---

# Contact

For questions or collaborations, please contact the HAIx Lab.
