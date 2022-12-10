# StylizedNeR Pytorch Implementation

## Introduction
The original paper is StylizedNeRF: Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D mutual learning.


## Installation

The code is tested with Google Colab, Python 3.7

Set Up Environment

    Run 'pip install -r requirements.txt' to install libraries


## Data preprocessing

    [1] Download the llff example data from official website http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
    [2] Prepare style images in ./style for stylized NeRF training and ./all_styles for VAE
  
## Pre-trained Model Preparation
    
    [1] Download the checkpoints of the VGG to ./pretrained
    [2] Train the decoder of AdaIN from scratch by running 'python train_style_modules.py --task finetune_decoder' or put the existing checkpoints of the decoder of AdaIN to ./pretrained
    [3] Run 'python train_style_modules.py --task vae' to pre-train the VAE
    
## Train and Evaluate a Stylized NeRF
    [1] Run 'python run_stylenerf.py --config ./configs/fern.txt' to train our model
    [2] Run 'python run_stylenerf.py --config ./configs/fern.txt --render_train_style --chunk 512' to check the outputs of the traning views
    [3] Run 'python run_stylenerf.py --config ./configs/fern.txt --render_valid_style --chunk 512' to check the outputs of the novel views

## Citation

Citation of the original code

    @inproceedings{Huang22StylizedNeRF,
        author = {Huang, Yi-Hua and He, Yue and Yuan, Yu-Jie and Lai, Yu-Kun and Gao, Lin},
        title = {StylizedNeRF: Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D Mutual Learning },
        booktitle={Computer Vision and Pattern Recognition (CVPR)},
        year = {2022},
    }
    
    @article{hu2020jittor,
      title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
      author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
      journal={Science China Information Sciences},
      volume={63},
      number={222103},
      pages={1--21},
      year={2020}
    }
