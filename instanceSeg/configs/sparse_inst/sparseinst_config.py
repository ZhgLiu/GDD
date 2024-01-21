_base_ = [
    '../_base_/datasets/coco_instance.py',  # 使用 COCO 实例分割数据集的配置
    '../_base_/schedules/schedule_1x.py',  # 使用标准的训练调度策略
    '../_base_/default_runtime.py'  # 默认的运行时设置
]

# 模型配置
model = dict(
    type='SparseInst',  # 模型类型为 SparseInst
    data_preprocessor=dict(
        type='DetDataPreprocessor',  # 数据预处理器类型
        # 其他数据预处理相关配置...
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    encoder=dict(
        # 编码器配置...
    ),
    decoder=dict(
        # 解码器配置...
    ),
    criterion=dict(
        # 训练时的匹配器和损失函数配置...
    ),
    test_cfg=dict(
        # 测试时的配置...
    )
)

# 训练和测试设置
train_cfg = dict(
    # 训练时的相关配置...
)
test_cfg = dict(
    # 测试时的相关配置...
)

# 优化器配置
optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# 学习率调度配置
lr_config = dict(policy='step', step=[8, 11])

# 运行器配置
runner = dict(type='EpochBasedRunner', max_epochs=12)
