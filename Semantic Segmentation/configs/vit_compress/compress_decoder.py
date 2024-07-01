_base_ = [
    "../_base_/models/fcn_vit_base.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
    "potsdam_dataset.py",
]

log_config = dict(
    interval=200,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.9,
    ),
)
checkpoint_config = dict(by_epoch=False, interval=160000)
lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

optimizer_config = dict(grad_clip=None)

model = dict(
    pretrained="path_to_pretrained_ckpt",
    freeze_backbone=True,
    backbone=dict(
        type="ViT_Compress_Entropy",
        img_size=512,
        patch_size=16,
        drop_path_rate=0.1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_abs_pos_emb=True,
        with_decoder=True,
    ),
    decode_head=dict(
        in_channels=512 // 2,
        channels=128,
        num_convs=2,
        num_classes=6,
    ),
    auxiliary_head=dict(
        in_channels=512 // 2,
        channels=64,
        num_convs=1,
        num_classes=6,
    ),
)
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
