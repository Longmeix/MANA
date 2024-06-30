# MANA [IPM, 2024]

The PyTorch codes and datasets for our paper "Locally-adaptive Mapping for Network Alignment via Meta-learning", which is published in Information Processing & Management 2024.

## 1. Requirements
- PyTorch >= 1.10.0+cu111
- Python >= 3.8.10
- networkx==2.6.1
- Numpy==1.20.1
- pandas==1.3.0
- scipy==1.6.2

The experiments were conducted on an Ubuntu machine equipped with four NVIDIA GeForce RTX 3090 GPUs. The CUDA version is 11.1+ , you can change the device into CPU if only CPU is available. 

## 2. Repository Structure
- datasets/: you need to put the prepared data here.  
    - FT/: `adj_s.pkl, adj_t.pkl` store adjacency matrices of the source network and the target network. `links_0.5.pkl` includes the training and tesing sets when training ratio=0.5. 
    - dataset/model/: `{Model}/{Dataset_Name}_0.5_best_embs.pkl` stores the mapped embeddings of two networks when getting the best performance. They are used for case study.
- prep_meta_tasks/: construct meta-train and meta-test tasks
- models/: common modules for network alignment
- utils/: tool functions for processing data and logging
- logs/: saving results
- config.py: hyperparameters
- run.py: run commands
- other folders/: baselines and baseline+MANA models

## 3. Running
```python
cd MANA

# mapping-based methods
python Node2vec/run_node2vec.py
python PALE/runPALE.py
python DeepLink/run_deeplink.py
python metaNA/runmetaNA.py
python GAEA/train_mana.py
python DegUIL/DegUIL_MANA.py

# Sharing-based methods
python dual_amn/main_dualamn.py
python JMAC_EA/main_jmac_na.py

```
Hyperparameter settings can refer to `run.py`, though the file may have been modified after getting the original results of our paper.

## 4. Citation
    @article{long2024mana,
    author       = {Meixiu Long, Siyuan Chen, and Jiahai Wang},
    title        = {Locally-adaptive Mapping for Network Alignment via Meta-learning},
    journal      = {Information Processing \& Management},
    volume       = {61},
    number       = {5},
    pages        = {103817},
    year         = {2024}
    }
