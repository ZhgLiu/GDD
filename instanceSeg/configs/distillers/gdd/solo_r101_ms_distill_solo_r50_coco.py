_base_ = [
    '../../solo/solo_r50_fpn_1x_coco.py'
]
# model settings
find_unused_parameters=True
alpha_gdd=200
temp_gdd=4
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'Solo_r101_3x.pth',
    init_student = True,
    distill_cfg = [ 
                    dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='fea_loss_gdd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_gdd=alpha_gdd,
                                       temp_gdd=temp_gdd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='fea_loss_gdd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_gdd=alpha_gdd,
                                       temp_gdd=temp_gdd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='fea_loss_gdd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_gdd=alpha_gdd,
                                       temp_gdd=temp_gdd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='fea_loss_gdd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       alpha_gdd=alpha_gdd,
                                       temp_gdd=temp_gdd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/solo/solo_r50_fpn_1x_coco.py'
teacher_cfg = 'configs/solo/solo_r101_fpn_3x_coco.py'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
