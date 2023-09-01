<h1 align = "center">
    Denoising Autoencoders for Improved Tabular Feature Representation
</h1>

## Overview

### Denoising AutoEncoders (DAE)
DAEs are [AutoEncoder](https://en.wikipedia.org/wiki/Autoencoder#:~:text=An%20autoencoder%20is%20a%20type,to%20ignore%20signal%20%E2%80%9Cnoise%E2%80%9D.) model trained to perform a denoising task. The model takes a partially corrupted input data and learns to clean and output the cleaned data.

Through the denoising task, the model learns the input distribution and produces latent representations that are robust to corruptions. The latent representations extracted from the model can be useful for a variety of downstream tasks. One can:  
    1. Freeze the encoder layers and use the latent representations to train supervised ML models, rendering DAE as a vehicle for automatic feature engineering.  
    2. Use the latent representations for unsupervised tasks like similarity query or clustering.  

### Applying Denoise AutoEncoder to Tabular data  
To train DAE on tabular data, the most important piece is the noise generator. What makes sense and most effective is [swap noise](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629), through which, each value in the training data maybe replaced by a random value from the same column.

### What's included
This package implements:  
    1. Swap Noise generator.  
    2. Dataframe parser which converts arbitrary pandas dataframe to numpy arrays.  
    3. Network constructor with configurable body blocks.  
    4. Training function.  
    5. Sklearn style `.fit`, `.transform` API.  
    6. Sklearn style model also supports `save` and `load`. 

## Installation

tabdae is built with PyTorch. Make sure to install the dependencies listed in [requirements.txt](./requirements.txt). Then install the package using pip:
```bash
pip install -r requirements.txt
pip install git+https://github.com/alexstedev/DenoisingAutoencoders.git
```

## Quickstart

```python
import pandas as pd
from tabdae.models.model import DAE

df = pd.read_csv(<path-to-csv-file>)

dae = DAE(
    body_network='deepstack',
    body_network_cfg=dict(hidden_size=1024),
    swap_noise_probas=.15,
    device='cuda',
)  

dae.fit(df, verbose=1, optimizer_params={'lr': 3e-4})

# extract latent representation with the model
latent = dae.transform(df)
```

## Credit  

While I haven't been able to find an article introducing this method, it has won several Kaggle competitions, e.g. [Porto Seguro's safe driver prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/discussion/44629) and [Tabular Playground Series - Feb 2021](https://www.kaggle.com/c/tabular-playground-series-feb-2021).

