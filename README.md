## [CVPR 2023] ViPLO - Official PyTorch implementation

![architecture_image](./docs/overall_arch.png)


**ViPLO: Vision Transformer based Pose-Conditioned Self-Loop Graph for Human-Object Interaction Detection**<br>
Jeeseung Park, Jin-Woo Park, Jong-Seok Lee<br>

Abstract: *Human-Object Interaction (HOI) detection, which localizes and infers relationships between human and objects, plays an important role in scene understanding. Although two-stage HOI detectors have advantages of high efficiency in training and inference, they suffer from lower performance than one-stage methods due to the old backbone networks and the lack of considerations for the HOI perception process of humans in the interaction classifiers. In this paper, we propose Vision Transformer based Pose-Conditioned Self-Loop Graph (ViPLO) to resolve these problems. First, we propose a novel feature extraction method suitable for the Vision Transformer backbone, called masking with overlapped area (MOA) module. The MOA module utilizes the overlapped area between each patch and the given region in the attention function, which addresses the quantization problem when using the Vision Transformer backbone. In addition, we design a graph with a pose-conditioned self-loop structure, which updates the human node encoding with local features of human joints. This allows the classifier to focus on specific human joints to effectively identify the type of interaction, which is motivated by the human perception process for HOI. As a result, ViPLO achieves the state-of-the-art results on two public benchmarks, especially obtaining a +2.07 mAP performance gain on the HICO-DET dataset.*

## Requirements

* We have done all testing and development using 4 RTX 3090 GPUs with 24GB.
* We recommend using Docker, with nvcr.io/nvidia/pytorch:21.06-py3 Image file. Please refer to [NVIDIA docs](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for more detail. 
* Additional python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`. 

## Installation

Step 1: Download the repository and its' submodules. `git clone https://github.com/Jeeseung-Park/ViPLO`, `git submodule update --init --recursive`

Step 2: Install the CLIP with our MOA module added via `pip install ./CLIP`

Step 3: Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)


## Preparing datasets

Pre-trained networks are stored as `*.pkl` files that can be referenced using local filenames

```.bash
# Generate images using pretrained_weight 
python generate.py --outdir=out --seeds=100-105 \
    --network=path_to_pkl_file
```

Outputs from the above commands are placed under `out/*.png`, controlled by `--outdir`. Downloaded network pickles are cached under `$HOME/.cache/dnnlib`, which can be overridden by setting the `DNNLIB_CACHE_DIR` environment variable. The default PyTorch extension build directory is `$HOME/.cache/torch_extensions`, which can be overridden by setting `TORCH_EXTENSIONS_DIR`.


## Preparing datasets (HICO-DET)

Step 1: Download the HICO-DET dataset. 
```.bash
bash hicodet/download.sh 
```

Step 2: Run a Faster R-CNN pre-trained on MS COCO to generate detections & Generate ground truth detections for test. 
```.bash
python hicodet/detections/preprocessing.py --partition train2015
python hicodet/detections/preprocessing.py --partition test2015
python hicodet/detections/generate_gt_detections.py --partition test2015 

```

Step 3: Generate fine-tuned detections from DETR-based detector, [UPT](https://github.com/fredzzhang/upt). 
```.bash

```


**CIFAR-10**: Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

**STL-10**: Download the stl-10 dataset 5k training, 100k unlabeled images from [STL-10 dataset page](https://cs.stanford.edu/~acoates/stl10/) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/~ --dest=~/datasets/stl10.zip \
    ---width=48 --height=48
```

**CelebA**: Download the CelebA dataset Aligned&Cropped Images from [CelebA dataset page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/~--dest=~/datasets/stl10.zip \
    ---width=64 --height=64
```


**LSUN Church**: Download the desired categories(church) from the [LSUN project page](https://www.yf.io/p/lsun/) and convert to ZIP archive:

```.bash

python dataset_tool.py --source=~/downloads/lsun/raw/church_lmdb --dest=~/datasets/lsunchurch.zip \
    --width=128 --height=128
```



## Training new networks

In its most basic form, training new networks boils down to:

```.bash
python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1 --batch=32 --cfg=cifar --g_dict=256,64,16 \
    --num_layers=1,2,2 --depth=32
```

* `--g_dict=` it means 'Hidden size' in paper, and it must be match with image resolution.
* `--num_layers=` it means 'Layers' in paper, and it must be match with image resolution.
* `--depth=32` it means minimum required depth is 32, described in Section 2 at paper.
* `--linformer=1` apply informer to Styleformer.

Please refer to [`python train.py --help`](./docs/train-help.txt) for the full list. 
To train STL-10 dataset with same setting at paper, please fix the starting resolution 8x8 to 12x12 at training/networks_Generator.py. 



## Quality metrics

Quality metrics can be computed after the training:

```.bash
# Pre-trained network pickle: specify dataset explicitly, print result to stdout.
python calc_metrics.py --metrics=fid50k_full --data=~/datasets/lsunchurch.zip \
    --network=path_to_pretrained_lsunchurch_pkl_file
    
python calc_metrics.py --metrics=is50k --data=~/datasets/lsunchurch.zip \
    --network=path_to_pretrained_lsunchurch_pkl_file    
```

## Citation
If you found our work useful, please don't forget to cite
```
@misc{park2021styleformer,
      title={Styleformer: Transformer based Generative Adversarial Networks with Style Vector}, 
      author={Jeeseung Park and Younggeun Kim},
      year={2021},
      eprint={2106.07023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



The code is heavily based on the [stylegan2-ada-pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch)
