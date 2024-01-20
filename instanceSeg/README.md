# Instance Segmentation
## Install
Please follow the installation instructions for MMDetection and ensure successful execution.
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html

pip install mmdetection==2.19.0
or
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git tag
git checkout v2.19.0
pip install -v -e .

pip install yapf==0.30.0

```
## Add and Replace the codes
```
#Add Folders
distillers/(ours) to configs/
mmdet/distillation/(ours) to mmdetection/mmdet/
```
```
#Replace files
mmdetection/mmdet/apis with mmdet/apis/train.py(ours)
mmdetection/tools/train.py with tools/train.py(ours)
```

## Train
```
#single GPU
python tools/train.py configs/distillers/gdd/solo_r101_ms_distill_solo_r50_coco.py


#multi GPU
bash tools/dist_train.sh configs/distillers/gdd/solo_r101_ms_distill_solo_r50_coco.py 2
