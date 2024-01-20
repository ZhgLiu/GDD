# GDD
Generative Denoise Distillation: Simple Stochastic Noises Induce Efficient Knowledge Transfer for Dense Prediction
![method](method.jpg)

The code is constantly updated

## Semantic Segmentation 
### Install
Please follow the installation instructions for MMSegmentation and ensure successful execution.

### Train
```
#single GPU
python tools/train.py configs/distillers/gdd/psp_r101_distill_psp_r18_40k_512x512_city.py

#multi GPU
bash tools/dist_train.sh configs/distillers/mgd/psp_r101_distill_psp_r18_40k_512x512_city.py 8
```
## Instance Segmentation
Please refer to [instance segmentation]()

## Object Classification
Please refer to [object detection]()

## Citation
```
@misc{liu2024generative,
      title={Generative Denoise Distillation: Simple Stochastic Noises Induce Efficient Knowledge Transfer for Dense Prediction}, 
      author={Zhaoge Liu and Xiaohao Xu and Yunkang Cao and Weiming Shen},
      year={2024},
      eprint={2401.08332},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
This code is based on [MGD](https://github.com/yzd-v/MGD), thanks for their contribution.
```
@article{yang2022masked,
  title={Masked Generative Distillation},
  author={Yang, Zhendong and Li, Zhe and Shao, Mingqi and Shi, Dachuan and Yuan, Zehuan and Yuan, Chun},
  journal={arXiv preprint arXiv:2205.01529},
  year={2022}
}
```