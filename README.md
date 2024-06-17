# Frustratingly Easy Test-Time Adaptation of Vision-Language Models
Code for the article "Frustratingly Easy Test-Time Adaptation of Vision-Language Models", arXiv, May 2024.

[![arXiv](https://img.shields.io/badge/arXiv-2405.18330-b31b1b.svg)](https://arxiv.org/abs/2405.18330)

Authors: [Matteo Farina](https://scholar.google.com/citations?user=SxQwDD8AAAAJ&hl=it&authuser=1), [Gianni Franchi](https://scholar.google.com/citations?hl=it&authuser=1&user=ZCW6-psAAAAJ), [Giovanni Iacca](https://scholar.google.com/citations?hl=it&authuser=1&user=qSw6YfcAAAAJ), [Massimiliano Mancini](https://scholar.google.com/citations?hl=it&authuser=1&user=bqTPA8kAAAAJ), [Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ&hl=it&authuser=1).

## Installation
### Dependencies
We provide both pip requirements and a conda environment to install the dependencies of this repository, feel free to choose the one that better suits your needs. The code was tested with **python 3.11.9**. 

Install pip requirements:
```
pip install -r requirements.txt
```

Install with conda:
```
conda env create -f environment.yaml
```

### Models
The only model weights you need to download are MaPLe's pretrained initializations. For your convenience, we provide a script to download them automatically. Simply run:
```
./scripts/download_maple.sh
```
You should now have a `weights` folder with the 3 MaPLe's ImageNet pretrainings provided by the authors (`weights/maple_seed1.pth`, `weights/maple_seed2.pth` and `weights/maple_seed3.pth`). Please check everything is in place. Should you have any problems, please download the weights from [this link](https://drive.google.com/drive/folders/18ISKsjc18e19Ov2nXOuH228FYBtgUa1O?usp=drive_link) and rename them accordingly. 

### Datasets
We strongly suggest you create a `datasets` folder under the root of this repository and store all datasets there.

#### Natural Distribution Shifts
For robustness to natural distribution shifts, we consider ImageNet-1k and 4 variants:  
1. [ImageNet-A](https://github.com/hendrycks/natural-adv-examples).
2. [ImageNet-v2](https://github.com/modestyachts/ImageNetV2) (we use the validation set of the `MatchedFrequency` version)
3. [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch).
4. [ImageNet-R](https://github.com/hendrycks/imagenet-r).

For all datasets simply download, extract and put them in the `./datasets` folder. You should have the following structure:
```
./datasets/
|  imagenet/
|  |  train/
|  |  |  # class folders
|  |  val/
|  |  |  # class folders

|  imagenet-a/
|  |  # class folders

|  imagenet-r/
|  |  # class folders

|  imagenet-sketch
|  |  # class folders

|  imagenetv2-matched-frequency-format-val  
|  | # class folders (0 to 999)
```

#### Finegrained Classification
For Finegrained classification, we adopt the same splits as [Zhou *et al*](https://arxiv.org/abs/2109.01134). Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets) for the installation of all datasets and the JSON files for the splits.
Once everything is downloaded, please organize everything as follows:  
```
./datasets/
|  caltech-101/
|  |  images/
|  |  |  # class folders
|  |  split_zhou_Caltech101.json

|  dtd/
|  |  images/
|  |  |  # class folders
|  |  split_zhou_DescribableTextures.json

|  fgvc_aircraft/
|  |  images/
|  |  |  # list of images
|  |  # a bunch of txt files

|  flower102/
|  |  jpg/
|  |  |  # list of images
|  |  split_zhou_OxfordFlowers.json

| food101/
|  |  images/
|  |  |  # class folders
|  |  split_zhou_Food101.json

|  oxford_pets/
|  |  images/
|  |  |  # list of images
|  |  split_zhou_OxfordPets.json

|  sun397/
|  | images/
|  |  |  # lettered folders ('a', 'b', 'c', etc.)
|  |  split_zhou_SUN397.json

|  ucf101/
|  |  images/
|  |  |  # class folders
|  |  split_zhou_UCF101.json
```

**IMPORTANT**. By the time of developing this work, the official Stanford Cars' website was unreachable. Please download images from [this Kaggle page](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) and annotations from [this Drive link](https://drive.google.com/drive/folders/13QnEkFQ8nhzf3jxo0RKX7UQAjrtAnYpR?usp=drive_link). You should organize files as follows:
```
./datasets/
|  stanford_cars/
|  |  images/
|  |  |  train/
|  |  |  |  # list of images
|  |  |  test/
|  |  |  |  # list of images
|  |  annots/
|  |  |  labels.csv
|  |  |  metadata.csv
|  |  |  split_coop.csv
```

## Run
The entrypoint for this repository is `run.py`. Please execute `python run.py --help` for a sense of the arguments.
We provide different bash files in `scripts` to run different versions of `Zero`:
1. `zero.sh` runs Vanilla `Zero`;
2. `zero_rlcf.sh` runs the `Zero` variant with a smaller CLIP-ViT-B-16 and a larger CLIP-ViT-L-14;

Note that the `--templates` flag activates the ensemble of textual templates (`+Ensemble` in Tab.1 and 2 of the article).
The `--maple` flag uses a MaPLe pretraining (only available with CLIP-ViT-B-16). 

## Citation
If you find this work useful, please consider citing: 
```
@article{farina2024frustratingly,
  title={Frustratingly Easy Test-Time Adaptation of Vision-Language Models},
  author={Farina, Matteo and Franchi, Gianni and Iacca, Giovanni and Mancini, Massimiliano and Ricci, Elisa},
  journal={arXiv preprint arXiv:2405.18330},
  year={2024}
}
```

## Acknowledgements
Parts of this repository are based on [TPT](https://github.com/azshue/TPT), [RLCF](https://github.com/mzhaoshuai/RLCF), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [CoOp](https://github.com/KaiyangZhou/CoOp) repositories. Huge thx to all authors!

## Contacts
Please do not hesitate to file an issue or to contact me at `m.farina@unitn.it`. I'll do my best to help!