_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/syn_ycbv_ID_low_mixed.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=11)))
workflow = [('train', 1),]
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# # actual epoch = 3 * 3 = 9
# lr_config = dict(policy='step', step=[3])
# # runtime settings
# runner = dict(
#     type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12
