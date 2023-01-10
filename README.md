# BERT for Acute Care Use Risk Prediction
***
## Name
"Natural Language Processing Methods to Identify Oncology Patients at High Risk for Acute Care with Clinical Notes" - ETH Master Thesis by Claudio Fanconi 

## Description
This repository is used for training deep language models in order to predict the risk of acute care utilization for patients that start chemotherapy. It is the code for the Paper [Natural Language Processing Methods to Identify Oncology Patients at High Risk for Acute Care with Clinical Notes](https://arxiv.org/pdf/2209.13860.pdf)

## Abstract
Clinical notes are an essential component of the health record. This paper evaluates how natural language processing (NLP) can be used to identify risk of acute care use (ACU) in oncology patients, once chemotherapy starts. Risk prediction using structured health data (SHD) is now standard, but predictions using free-text formats are complex. This paper explores the use of free-text notes for prediction of ACU in leu of SHD. Deep Learning models were compared to manually engineered language features. Results show that SHD models minimally outperform NLP models; an `1-penalised logistic regression with SHD achieved a C-statistic of 0.748 (95%-CI: 0.735, 0.762), while the same model with language features achieved 0.730 (95%-CI: 0.717, 0.745) and a transformer-based model achieved 0.702 (95%-CI: 0.688, 0.717). This paper shows how language models can be used in clinical applications and underlines how risk bias is different for diverse patient groups, even using only free-text data.
## Cite Us

```
@article{fanconi2022acu_nlp,
    title={Natural Language Processing Methods to Identify Oncology Patients at High Risk for Acute Care with Clinical Notes}, 
    author={Claudio Fanconi and Marieke van Buchem and Tina Hernandez-Boussard},
    year={2022},
    booktitle={AMIA 2023 Informatics Summit},
}
```


## Installation
Clone the current repository
```
git clone https://github.com/su-boussard-lab/bert-for-acu
cd bert-for-acu
```

I suggest to create a virtual environment and install the required packages.
```
conda create --name acu python=3.8
conda activate acu
conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia
conda install -r requirements.txt
```

### Source Code Directory Tree
```
.
├── experiment_scripts  # Config files to rerun training of the BERT models
└── src                 # Source code            
    ├── layers              # Single Neural Network layers
    ├── model               # Neural Network Models for NLP
    ├── data                # Folder with data processing parts
    └── utils               # Useful functions, such as loggers and config 
```

## Running the Experiments
To run the models, you first need to prepare the data. For this experiment we expect five CSV files: `TEXTS.csv` shall contain the unstructured health notes, `SHR.csv` should contain the structured health records, `LABELS.csv` should contain the labels. Both of these should be indexed by a patient deidentifier number. `TEST_IDS.csv` and `TRAIN_IDS.csv` are CSV files that contain the patiend deid files of the test and training set, respectively. You can change the paths in `config.yml` file. In this file you can also set which model should be fitted, by setting their flags to either True or False

The configs used for the experiments can be found here:
- language: `experiment_scripts/language_BERT.yml`
- fusion: `experiment_scripts/fusion_BERT.yml`
If you wish to have the trained weights directly, please reach out to fanconic@ethz.ch

To train the model on one single GPU, you can execute the script
```
bash run_experiment.sh
```

To run the model on multiple GPU, using data parallelism, you can run the script
```
bash run_parallel_experiment.sh
```


## Authors
- Claudio Fanconi (fanconic@ethz.ch) (code)
- Marieke van Buchem
- Tina Hernandez-Boussard



## Project status
Completed
