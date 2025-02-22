# On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning


## Credits
Our implementation is based on [ALBEF](https://github.com/salesforce/ALBEF) and [iSogCLR](https://github.com/zhqiu/contrastive-learning-iSogCLR/tree/main/bimodal_exps).

## Data

Please download the CC3M dataset for bimodal pretraining from [[link]](https://ai.google.com/research/ConceptualCaptions/download) and modify the `--data_path` and `--train_image_root` arguments of `clip.py` accordingly. It also needs the annotations `clip_train` that can be downloaded from [[link]](https://drive.google.com/drive/folders/1hAd0956xIztfwq0WrWLTGBx8sNuye595?usp=sharing).

Please also download the val and test splits of datasets [MS-COCO](https://cocodataset.org/#download), and [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/). The Cifar100 dataset can be downloaded by Torchvision datasets and the val split of ImageNet1k can be downloaded from, e.g., [[link]](https://huggingface.co/datasets/mrm8488/ImageNet1K-val). 

After downloading the data, one needs to organize the data folder as follows:
```
.
+--cc3m
+--coco
|  +--val2014
|  +--test2015
|
+--flickr30k
|  +--flickr30k_images
|
+--clip_train 
```

## Pretraining

Please check the `./scripts` folder for examples of training scripts. We also provide the logs of running NUCLR and SogCLR using the corresponding training scripts.

## Evaluation

You can evaluate the pretrained model using the `--evaluate` argument of `clip.py` and specifying the path to the saved checkpoint. We provide an example `run_evaluation.slurm` in the `scripts` folder.

## More Details of Implementation


The NUCLR algorithm is implemented in "./models/losses.py" as "DGCL_Loss". Consider a compositional function $L(\mathbf{w})=f(\ell(\mathbf{w}))$, the gradient is $\nabla L(\mathbf{w})=f'(\ell(\mathbf{w}))\nabla \ell(\mathbf{w})$. "DGCL_Loss" implements the “half-loss-half-gradient” $\text{detach}[f'(\ell(\mathbf{w}))] \ell(\mathbf{w})$  such that the gradient $\nabla L(\mathbf{w})$ can can be computed via autodiff on $\text{detach}[f'(\ell(\mathbf{w}))] \ell(\mathbf{w})$. Please refer to [this note](./dpm_grad_comp.pdf) for the detailed derivation. 


## How to cite
If you find our work useful, please consider citing our paper

```
 @inproceedings{wang2025nuclr,
    title={On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning},
    author={Wang, Bokun and Lei, Yunwen and Ying, Yiming and Yang, Tianbao},
    journal={International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=s15HrqCqbr}
}
```
