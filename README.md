<h1 align="center"><ins>Su</ins>bgraph-based Graph <ins>Re</ins>presentation <ins>L</ins>earning Framework</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2202.13538"><img src="https://img.shields.io/badge/-Paper-grey?logo=read%20the%20docs&logoColor=green" alt="Paper"></a>
    <a href="https://ogb.stanford.edu/docs/leader_linkprop/"><img src="https://img.shields.io/badge/OGB-LinkPred-blue" alt="OGBL"></a>
    <a href="https://github.com/Graph-COM/SUREL/tree/main/surel_gacc"><img src="https://img.shields.io/badge/GACC-v1.0-orange" alt="Version"></a>
    <a href="https://github.com/Graph-COM/SUREL"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
</p>

SUREL is a novel walk-based computation framework for efficient large-scale graph representation learning via subgraphs (SGRL).
Details on how SUREL works can be found in our paper [Algorithm and System Co-design for Efficient Subgraph-based Graph Representation Learning](https://arxiv.org/pdf/2202.13538.pdf), to appear in VLDB 2022.

Currently, we support:
- Large-scale graph learning tasks: link prediction / relation prediction / higher-order pattern prediction
- Preprocessing and training of datasets in OGB format
- Python API for user defined sampling and joining procedures
- Single GPU training and evaluation
- Relative Position Encoding + Node Features

We are working on expanding the functionality of SUREL to include:
- Multi-GPU training
- BrainVessel Dataset

## Requirements ##
(Other versions may work, but are untested)
* Ubuntu 20.04
* CUDA >= 10.2
* python >= 3.8
* 1.8 <= pytorch < 1.10 
* gcc >= 8.4
* cmake >= 3.16
* make >= 4.2

## SGRL Environment Setup ##

Requirements: Python >= 3.8, [Anaconda3](https://www.anaconda.com/)

- Update conda:
```bash
conda update -n base -c defaults conda
```

- Install basic dependencies to virtual environment and activate it: 
```bash
conda env create -f environment.yml
conda activate sgrl-env
```

- Example commends of installation for PyTorch (>= 1.8.0) and torch-geometric (>=1.6.0) with CUDA 10.2:
```bash
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
pip install torch-geometric==1.6.3
```
For more details, please refer to the [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/1.6.3/notes/installation.html). The code of this repository is lately tested with Python 3.8.5 + PyTorch 1.8.0 + torch-geometric 1.6.3.

## Quick Start

1. Install required version of PyTorch that is compatible with your CUDA driver

2. Clone the repository `git clone https://github.com/VeritasYin/SUREL.git`

3. Build and install SUREL_GAcc `cd surel_gacc;python3 setup.py install`

- To train **SUREL** for link prediction on Collab:
```bash
python main.py --dataset ogbl-collab --metric hit --num_step 4 --num_walk 200 --use_val
```

- To train **SUREL** for link prediction on Citation2:
```bash
python main.py --dataset ogbl-citation2 --metric mrr --num_step 4 --num_walk 100
```

- To train **SUREL** for relation prediction on MAG(A-P):
```bash
python main_hetro.py --dataset mag --relation write --metric mrr --num_step 3 --num_walk 100 --k 10
```

- To train **SUREL** for higher-order prediction on DBLP:
```bash
python main_horder.py --dataset DBLP-coauthor --relation cite --metric mrr --num_step 3 --num_walk 100
```

- All detailed training logs can be found at `<log_dir>/<dataset>/<training-time>.log`. For each instance, one-line summary will be reported to `<log_dir>/result_summary.log` after model training.

## Usage Summary
```
usage: Interface for SUREL framework [-h]
                                     [--dataset {ogbl-ppa,ogbl-citation2,ogbl-collab,mag,DBLP-coauthor,tags-math}]
                                     [--model {RNN,MLP,Transformer,GNN}]
                                     [--layers LAYERS]
                                     [--hidden_dim HIDDEN_DIM] [--x_dim X_DIM]
                                     [--data_usage DATA_USAGE]
                                     [--train_ratio TRAIN_RATIO]
                                     [--valid_ratio VALID_RATIO]
                                     [--test_ratio TEST_RATIO]
                                     [--metric {auc,mrr,hit}] [--seed SEED]
                                     [--gpu_id GPU_ID] [--nthread NTHREAD]
                                     [--B_size B_SIZE] [--num_walk NUM_WALK]
                                     [--num_step NUM_STEP] [--k K]
                                     [--directed DIRECTED] [--use_feature]
                                     [--use_weight] [--use_degree]
                                     [--use_htype] [--use_val] [--norm NORM]
                                     [--optim OPTIM] [--rtest RTEST]
                                     [--eval_steps EVAL_STEPS]
                                     [--batch_size BATCH_SIZE]
                                     [--batch_num BATCH_NUM] [--lr LR]
                                     [--dropout DROPOUT] [--l2 L2]
                                     [--patience PATIENCE] [--repeat REPEAT]
                                     [--log_dir LOG_DIR] [--res_dir RES_DIR]
                                     [--stamp STAMP]
                                     [--summary_file SUMMARY_FILE] [--debug]
                                     [--abs] [--save] [--load_dict]
                                     [--load_model] [--memo MEMO]
```

## Optional Arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --dataset {mag}       dataset name
  --relation {write,cite}
                        relation type
  --model {RNN,MLP,Transformer,GNN}
                        base model to use
  --layers LAYERS       number of layers
  --hidden_dim HIDDEN_DIM
                        hidden dimension
  --x_dim X_DIM         dim of raw node features
  --data_usage DATA_USAGE
                        use partial dataset
  --train_ratio TRAIN_RATIO
                        mask partial edges for training
  --valid_ratio VALID_RATIO
                        use partial valid set
  --test_ratio TEST_RATIO
                        use partial test set
  --metric {auc,mrr,hit}
                        metric for evaluating performance
  --seed SEED           seed to initialize all the random modules
  --gpu_id GPU_ID       gpu id
  --nthread NTHREAD     number of thread
  --B_size B_SIZE       set size of train sampling
  --num_walk NUM_WALK   total number of random walks
  --num_step NUM_STEP   total steps of random walk
  --k K                 number of paired negative queries
  --directed DIRECTED   whether to treat the graph as directed
  --use_feature         whether to use raw features as input
  --use_weight          whether to use edge weight as input
  --use_degree          whether to use node degree as input
  --use_htype           whether to use node type as input
  --use_val             whether to use val as input
  --norm NORM           method of normalization
  --optim OPTIM         optimizer to use
  --rtest RTEST         step start to test
  --eval_steps EVAL_STEPS
                        number of steps to test
  --batch_size BATCH_SIZE
                        mini-batch size (train)
  --batch_num BATCH_NUM
                        mini-batch size (test)
  --lr LR               learning rate
  --dropout DROPOUT     dropout rate
  --l2 L2               l2 regularization (weight decay)
  --patience PATIENCE   early stopping steps
  --repeat REPEAT       number of training instances to repeat
  --log_dir LOG_DIR     log directory
  --res_dir RES_DIR     resource directory
  --stamp STAMP         time stamp
  --summary_file SUMMARY_FILE
                        brief summary of training results
  --debug               whether to use debug mode
  --save                whether to save RPE to files
  --load_dict           whether to load RPE from files
  --load_model          whether to load saved model from files
  --memo MEMO           notes
```

## Citation
Please cite our paper if you are interested in our work.
```
@article{yin2022algorithm,
  title={Algorithm and System Co-design for Efficient Subgraph-based Graph Representation Learning},
  author={Yin, Haoteng and Zhang, Muhan and Wang, Yanbang and Wang, Jianguo and Li, Pan},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  year={2022}
}
```
