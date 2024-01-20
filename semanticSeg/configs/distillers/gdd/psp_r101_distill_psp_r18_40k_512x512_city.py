_base_ = [
    '../../pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
alpha_gdd=145
temp_gdd=4
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
    init_student = False,
    use_logit = False,
    distill_cfg = [ dict(methods=[dict(type='FeatureLoss',
                                       name='fea_loss_gdd',
                                       student_channels = 512,
                                       teacher_channels = 2048,
                                       alpha_gdd=alpha_gdd,
                                       temp_gdd=temp_gdd,
                                       )
                                ]
                        ),
                   ]
    )

student_cfg = 'configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py'
