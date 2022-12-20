# Uncertainty Estimation of Acute Care Utilization with Deep Neural Networks using Free-text Clinical Notes
***
## Name
"Uncertainty Estimation of Acute Care Utilization with Deep Neural Networks using Free-text Clinical Notes" - ETH Master Thesis by Claudio Fanconi 

## Description
This repository is used for training deep language models in order to predict the risk of acute care utilization for patients that start chemotherapy.


## Installation
Clone the current repository
```
git clone https://code.stanford.edu/fanconic/claudio-master-thesis
cd claudio-master-thesis
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
└── src                 # Source code            
    ├── layers              # Single Neural Network layers
    ├── model               # Neural Network Models for NLP
    ├── data                # Folder with data processing parts
    └── utils               # Useful functions, such as loggers and config 
```


## Authors
Claudio Fanconi (fanconic@stanford.edu, fanconic@ethz.ch, fanconic@gmail.com)


## Project status
Completed
