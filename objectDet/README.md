# Semantic Segmentation 
## Install
Please follow the installation instructions for MMSegmentation and ensure successful execution.
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html

pip install mmsegmentation==0.20.2
or
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git tag
git checkout v0.20.2
pip install -v -e .

pip install yapf==0.30.0

```
## Add and Replace the codes
```
#Add Files
pspnet_r18-d8_512x512_40k_cityscapes.py(ours) to configs/pspnet/
deeplabv3_r18-d8_512x512_40k_cityscapes.py(ours) to configs/deeplabv3/
cityscapes_512x512.py(ours) to configs/_base_/datasets/
```
```
#Add Folders
distillers/(ours) to configs/
mmseg/distillation/(ours) to mmsegmentation/mmseg/
```
```
#Replace files
mmsegmenation/mmseg/apis with mmseg/apis/train.py(ours)
mmsegmentation/tools/train.py with tools/train.py(ours)
```

## Train
```
#single GPU
python tools/train.py configs/distillers/gdd/psp_r101_distill_psp_r18_40k_512x512_city.py
python tools/train.py configs/distillers/gdd/psp_r101_distill_deepv3_r18_40k_512x512_city.py


#multi GPU
bash tools/dist_train.sh configs/distillers/gdd/psp_r101_distill_psp_r18_40k_512x512_city.py 2
bash tools/dist_train.sh configs/distillers/gdd/psp_r101_distill_deepv3_r18_40k_512x512_city.py 2