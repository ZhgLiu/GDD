_base_ = [
    '../../sparse_inst/sparseinst_r18_fpn_1x_coco.py'
]

# model settings
find_unused_parameters=True
alpha_gdd=300
temp_gdd=4
# 蒸馏器配置
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained='path/to/sparseinst_resnet50_pretrained.pth',  # 教师模型预训练权重
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

# 学生模型配置文件路径
student_cfg = 'configs/sparse_inst/sparseinst_r18_fpn_1x_coco.py'
# 教师模型配置文件路径
teacher_cfg = 'configs/sparse_inst/sparseinst_r50_fpn_1x_coco.py'

# 优化器配置
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))