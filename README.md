<h1 align="center">SUREL: <ins>Su</ins>bgraph-based Graph <ins>Re</ins>presentation <ins>L</ins>earning Framework</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2202.13538"><img src="https://img.shields.io/badge/-Paper-grey?logo=read%20the%20docs&logoColor=green" alt="Paper"></a>
    <a href="https://github.com/Graph-COM/SUREL"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
    <a href="https://github.com/Graph-COM/SUREL/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD%202--Clause-red.svg"></a>
    <a href="https://ogb.stanford.edu/docs/leader_linkprop/"><img src="https://img.shields.io/badge/OGB-LinkPred-blue" alt="OGBL"></a>
    <a href="https://github.com/Graph-COM/SUREL/tree/main/surel_gacc"><img src="https://img.shields.io/badge/GACC-v1.0-orange" alt="Version"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGraph-COM%2FSUREL&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits&edge_flat=false"/></a>
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
python main_horder.py --dataset DBLP-coauthor --metric mrr --num_step 3 --num_walk 100
```

- All detailed training logs can be found at `<log_dir>/<dataset>/<training-time>.log`.

## Result Reproduction
This section supplements our SUREL paper accepted in VLDB 2022. To reproduce the results of SUREL reported in Tables 3 and 4, use the following command:
* OGBL - Link Prediction
```bash
python3 main.py --dataset <dataset> --metric <metric> --num_step <num_step> --num_walk <num_walk> --k <k>
```
where 'dataset' can be either of 'ogbl-citation2', 'ogbl-collab' and 'ogbl-ppa'; 'metric' can be either 'mrr' or 'hit'.
* Relation Prediction
```bash
python main_hetro.py --dataset mag --relation <relation> --metric mrr --num_step <num_step> --num_walk <num_walk> --k <k>
```
where 'relation' can be either 'write' or 'cite'. 
* Higher-order Prediction
```bash
python main_horder.py --dataset <dataset> --metric mrr --num_step <num_step> --num_walk <num_walk> --k <k>
```
where 'dataset' can be either 'DBLP-coauthor' or 'tags-math'.

The detailed parameter configurations are provided in Table 8, Appendix D of the [arxiv version](https://arxiv.org/abs/2202.13538) of this work. For the profiling of SUREL in Table 4 and Fig. 4 (a-b), please use the parameter setting provided in Appendix D.3. 

To test the scaling performance of Walk Sampler and RPE Joining, functions 'run_walk' and 'sjoin' can be imported and called from the module 'surel_gacc'. Please adjust the parameter values of 'num_walk', 'num_step' and 'nthread' accordingly as Fig. 4 (c-d) shown.

To perform hyper-parameter analysis of the number of walks 𝑀, the step of walks 𝑚, and the hidden dimension 𝑑, please adjust the parameter values of 'num_walk', 'num_step' and 'hidden_dim' accordingly as Fig. 5 shown. 

<details>
  <summary>Sample Output</summary>
  
```text
2022-03-25 15:57:16,677 - root - INFO - Create log file at ./log/ogbl-citation2/032522_155716.log
2022-03-25 15:57:16,677 - root - INFO - Command line executed: python main.py --gpu 2 --patience 5 --hidden_dim 64 --seed 0
2022-03-25 15:57:16,677 - root - INFO - Full args parsed:
2022-03-25 15:57:16,677 - root - INFO - Namespace(B_size=1500, batch_num=2000, batch_size=32, data_usage=1.0, dataset='ogbl-citation2', debug=False, directed=False, dropout=0.1, eval_steps=100, gpu_id=2, hidden_dim=64, k=50, l2=0.0, layers=2, load_dict=False, load_model=False, log_dir='./log/', lr=0.001, memo=None, metric='mrr', model='RNN', norm='all', nthread=16, num_step=4, num_walk=100, optim='adam', patience=5, repeat=1, res_dir='./dataset/save', rtest=499, save=False, seed=0, stamp='032522_155716', summary_file='result_summary.log', test_ratio=1.0, train_ratio=0.05, use_degree=False, use_feature=False, use_htype=False, use_val=False, use_weight=False, valid_ratio=0.1, x_dim=0)
2022-03-25 15:57:16,727 - root - INFO - torch num_threads 16
2022-03-25 15:57:26,536 - root - INFO - eval metric                                                            mrr
task type                                                  link prediction
download_name                                                  citation-v2
version                                                                  1
url                      http://snap.stanford.edu/ogb/data/linkproppred...
add_inverse_edge                                                     False
has_node_attr                                                         True
has_edge_attr                                                        False
split                                                                 time
additional node files                                            node_year
additional edge files                                                 None
is hetero                                                            False
binary                                                               False
Name: ogbl-citation2, dtype: object
Keys: ['x', 'edge_index', 'node_year']
2022-03-25 15:57:26,536 - root - INFO - node size 2927963, feature dim 128, edge size 30387995 with mask ratio 0.05
2022-03-25 15:57:26,536 - root - INFO - use_weight False, use_coalesce False, use_degree False, use_val False
2022-03-25 15:57:45,775 - root - INFO - Sparsity of loaded graph 6.727197221716796e-06
2022-03-25 15:57:45,782 - root - INFO - Observed subgraph with 2918932 nodes and 28836021 edges;
2022-03-25 15:57:45,789 - root - INFO - Training subgraph with 1394162 nodes and 1519315 edges.
2022-03-25 15:57:50,400 - root - INFO - #Model Params 79617
2022-03-25 15:59:14,643 - root - INFO - Samples: valid 8659 by 1000 test 86596 by 1000 metric: mrr
2022-03-25 15:59:15,405 - root - INFO - Running Round 1
2022-03-25 15:59:29,229 - root - INFO - Batch 1	W1502/D1394162	Loss: 0.1971, AUC: 0.5049
2022-03-25 15:59:42,266 - root - INFO - Batch 2	W2991/D1394162	Loss: 0.1097, AUC: 0.4975
2022-03-25 15:59:56,187 - root - INFO - Batch 3	W4431/D1394162	Loss: 0.1024, AUC: 0.4976
2022-03-25 16:00:09,070 - root - INFO - Batch 4	W5761/D1394162	Loss: 0.1030, AUC: 0.4980
2022-03-25 16:00:23,285 - root - INFO - Batch 5	W7215/D1394162	Loss: 0.1013, AUC: 0.5053
...
```
</details>

## Usage
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

<details>
  <summary>Optional Arguments</summary>

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
</details>

## Citation
Please cite our paper if you are interested in our work.
```
@article{yin2022algorithm,
  title={Algorithm and System Co-design for Efficient Subgraph-based Graph Representation Learning},
  author={Yin, Haoteng and Zhang, Muhan and Wang, Yanbang and Wang, Jianguo and Li, Pan},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={11},
  pages={2788-2796},
  year={2022}
}
```
