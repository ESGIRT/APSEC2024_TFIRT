<p align="center">
<img src="https://github.com/ESGIRT/ESEM2024-TransformerIRT/blob/main/Image/logo.png" alt="TAIRT" width="50%" />
</p>


 ## About
This repository is the replication package for paper: **"An Empirical Study of Transformer Models on Automatically Templating GitHub Issue Reports"**.

## Content
The replication package contains the datasets, code, and models used in our study.
- **Model:** This folder holds the pre-trained models based on the Transformer architecture used in our experiments: `BERT`, `RoBERTa`, `GPT2`, `BART`, `T5`. Researchers can change the configuration of the models in the code to meet the needs of their own research.
- **Dataset:** Stored dataset: `autoirt.csv`.
- **LLM:** The code for directly invoking the interfaces of the large models `GPT3.5` and `Claude3` for automatic templating tasks. It is important to note that researchers need to input their own *API key* in the code.

## Overview
The following diagram summarizes our work.
<p align="center">
<img src="https://github.com/ESGIRT/ESEM2024-TransformerIRT/blob/main/Image/overview.png" alt="TAIRT" width="80%" />
</p>
