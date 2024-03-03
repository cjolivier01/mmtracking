
# Stitched panoramic video aspect ratio is about 2.4
#detection_img_scale = (1333, 800)
#detection_img_scale = (2600, 1600)
#detection_img_scale = (1920, 800)
detection_img_scale = (1680, 700)

dcn_detector = dict(
    type="FasterRCNN",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        dcn=dict(type="DCNv2", deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(
        type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
        ),
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint="/home/colivier/Downloads/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth",
    ),
)

#
# The overall tracking model
#
model = dict(
    detector=dcn_detector,
    type="ByteTrack",
    motion=dict(type="KalmanFilter"),
    tracker=dict(
        type="ByteTracker",
        # obj_score_thrs=dict(high=0.6, low=0.1),
        # init_track_thr=0.7,
        obj_score_thrs=dict(high=0.4, low=0.1),
        init_track_thr=0.3,
        # obj_score_thrs=dict(high=0.99, low=0.99),
        # init_track_thr=0.99,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30,
    ),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(type="Resize", img_scale=detection_img_scale, keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
    val=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=detection_img_scale,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/instances_val2017.json",
        img_prefix="data/coco/val2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=detection_img_scale,
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)


# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type="MOTChallengeDataset",
#         visibility_thr=-1,
#         ann_file="data/MOT17/annotations/train_cocoformat.json",
#         img_prefix="data/MOT17/train",
#         ref_img_sampler=dict(
#             num_ref_imgs=1, frame_range=10, filter_key_img=True, method="uniform"
#         ),
#         pipeline=[
#             {"type": "LoadMultiImagesFromFile", "to_float32": True},
#             {"type": "SeqLoadAnnotations", "with_bbox": True, "with_track": True},
#             {
#                 "type": "SeqResize",
#                 "img_scale": (1088, 1088),
#                 "share_params": True,
#                 "ratio_range": (0.8, 1.2),
#                 "keep_ratio": True,
#                 "bbox_clip_border": False,
#             },
#             {"type": "SeqPhotoMetricDistortion", "share_params": True},
#             {
#                 "type": "SeqRandomCrop",
#                 "share_params": False,
#                 "crop_size": (1088, 1088),
#                 "bbox_clip_border": False,
#             },
#             {"type": "SeqRandomFlip", "share_params": True, "flip_ratio": 0.5},
#             {
#                 "type": "SeqNormalize",
#                 "mean": [123.675, 116.28, 103.53],
#                 "std": [58.395, 57.12, 57.375],
#                 "to_rgb": True,
#             },
#             {"type": "SeqPad", "size_divisor": 32},
#             {"type": "MatchInstances", "skip_nomatch": True},
#             {
#                 "type": "VideoCollect",
#                 "keys": [
#                     "img",
#                     "gt_bboxes",
#                     "gt_labels",
#                     "gt_match_indices",
#                     "gt_instance_ids",
#                 ],
#             },
#             {"type": "SeqDefaultFormatBundle", "ref_prefix": "ref"},
#         ],
#     ),
#     val=dict(
#         type="MOTChallengeDataset",
#         ann_file="data/MOT17/annotations/train_cocoformat.json",
#         img_prefix="data/MOT17/train",
#         ref_img_sampler=None,
#         pipeline=[
#             {"type": "LoadImageFromFile"},
#             {
#                 "type": "MultiScaleFlipAug",
#                 "img_scale": (1088, 1088),
#                 "flip": False,
#                 "transforms": [
#                     {"type": "Resize", "keep_ratio": True},
#                     {"type": "RandomFlip"},
#                     {
#                         "type": "Normalize",
#                         "mean": [123.675, 116.28, 103.53],
#                         "std": [58.395, 57.12, 57.375],
#                         "to_rgb": True,
#                     },
#                     {"type": "Pad", "size_divisor": 32},
#                     {"type": "ImageToTensor", "keys": ["img"]},
#                     {"type": "VideoCollect", "keys": ["img"]},
#                 ],
#             },
#         ],
#     ),
#     test=dict(
#         type="MOTChallengeDataset",
#         ann_file="data/MOT17/annotations/train_cocoformat.json",
#         img_prefix="data/MOT17/train",
#         ref_img_sampler=None,
#         pipeline=[
#             {"type": "LoadImageFromFile"},
#             {
#                 "type": "MultiScaleFlipAug",
#                 "img_scale": (1088, 1088),
#                 "flip": False,
#                 "transforms": [
#                     {"type": "Resize", "keep_ratio": True},
#                     {"type": "RandomFlip"},
#                     {
#                         "type": "Normalize",
#                         "mean": [123.675, 116.28, 103.53],
#                         "std": [58.395, 57.12, 57.375],
#                         "to_rgb": True,
#                     },
#                     {"type": "Pad", "size_divisor": 32},
#                     {"type": "ImageToTensor", "keys": ["img"]},
#                     {"type": "VideoCollect", "keys": ["img"]},
#                 ],
#             },
#         ],
#     ),
# )
