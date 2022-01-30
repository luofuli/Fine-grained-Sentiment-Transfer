# Fine-grained Text Sentiment Transfer
This repository contains the original implementation of the models presented in
[Towards Fine-grained Text Sentiment Transfer](https://www.aclweb.org/anthology/P19-1194.pdf) (ACL 2019).

<img width="220" height="220" src="image/example.jpg"></img>
<img width="400" height="250" src="image/Seq2SentiSeq.jpg"></img>
<img width="230" height="150" src="image/CycleRL.jpg"></img>

## Reproducibility
**In order to help you quickly reproduce the existing works of fine-grained text sentiment transfer, we release the outputs of all models and the corresponding references.**

- **Outputs**: Generated results (outputs) of **3 baselines** and **our model** are in the `outputs/` directory.
- **References**: Human references are in the `data/yelp/reference.txt` file.

## Dependencies
```
python==2.7
numpy==1.14.2
tensorflow==1.13.1
(You might need to install tensorflow-gpu as well)
Also even after installing tensorflow if it does not reflect in your virtual env it might be due to pip install issues so check the installation location using the appropriate commands
OpenNMT-tf==1.15.0 
```

## Data Formatting for Custom Dataset

All the spaces in the different datafiles are tab spaces and not normal spaces and using normal spaces cause errors

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

Incase of any errors in this step due to YAML loader constructor namely this error - ```yaml.constructor.ConstructorError: could not determine a constructor for the tag 'tag:yaml.org,2002:python/object:argparse.Namespace'``` changing the load_args_from_yaml() function inside common_options.py to 
```
def load_args_from_yaml(dir):
    args = load(open(os.path.join(dir, 'conf.yaml')), Loader=UnsafeLoader)
    return args
```
while importing UnsafeLoader from yaml might be a workaround. [Reference](https://github.com/yaml/pyyaml/issues/482#issuecomment-765607132)

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
