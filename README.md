# Fine-grained Text Sentiment Transfer
This repository contains the original implementation of the models presented in
[Towards Fine-grained Text Sentiment Transfer](https://www.aclweb.org/anthology/P19-1194.pdf) (ACL 2019).

## Dependencies
```
python==2.7
numpy==1.14.2
tensorflow==1.13.1
OpenNMT-tf==1.15.0 
```

## Quick Start

### Step 1: Pre-train the sentiment scorer
A pretrained sentiment scorer is used to compute the sentiment transformation reward. Here the scorer is implemented as LSTM-based linear regression model. You can train the model using the following command:
```
cd regressor/
python main.py --mode train
```
**Note:** If you get the error `no module named opennmt`, please install `OpenNMT-tf`: `pip install OpenNMT-tf==1.15.0`.

### Step 2: Pre-train the Seq2SentiSeq model using pseudo-parallel data
You can train the Seq2SentiSeq model using the following command:
```
cd seq2sentiseq/
python main.py --mode train
```

### Step 3: Cycle reinforcement learning
After finishing the previous two steps, you can start the cycle reinforcement learning using the following command:
```
python cycle_training.py --n_epoch 30
```
The final transffered results are in the `../tmp/output/yelp_final_*/` dir.

## Cite
Please cite the following paper if you found the resources in this repository useful.
```
@inproceedings{luo2019towards,
  title={Towards Fine-grained Text Sentiment Transfer},
  author={Luo, Fuli and Li, Peng and Yang, Pengcheng and Zhou, Jie and Tan, Yutong and Chang, Baobao and Sui, Zhifang and Sun, Xu},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, {ACL} 2019},
  pages={2013--2022},
  year={2019}
}
```
