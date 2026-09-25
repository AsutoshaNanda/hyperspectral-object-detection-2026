# Architecture adapter sources

Verified on 2026-09-19 against official project documentation or source. These are implementation and licensing records, not competition results.

| IDs | Official source | What was verified |
|---|---|---|
| CR1 | https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn | Current MMDetection Cascade R-CNN R50-FPN config and MMDetection 3.x build/inference APIs. Torchvision has no Cascade R-CNN implementation. |
| DF1, DF2, E1b, E1c | https://github.com/Peterande/D-FINE | Official S/M configs and checkpoints, Apache-2.0 code, full `DFINETransformer`, Fine-grained Distribution Refinement outputs, `loss_fgl`, and GO-LSD `loss_ddf`. Objects365-derived checkpoints have separate dataset-term risk, so the catalog selects COCO checkpoints for the first smoke. |
| DF3 | https://github.com/Peterande/D-FINE/blob/master/src/zoo/dfine/dfine_decoder.py | FDR changes iterative distribution regression inside the decoder. A full D-FINE replacement is rejected as an isolated RT-DETR regression-head ablation. |
| E1a | https://github.com/lyuwenyu/RT-DETR and https://huggingface.co/PekingU/rtdetr_v2_r18vd | Official RT-DETRv2 project and public Transformers checkpoint/model card. |
| E1d | https://github.com/IDEA-Research/DINO | Official DINO Swin detector configs and public model-zoo checkpoints. |
| E1e | https://github.com/Sense-X/Co-DETR | Official Co-DINO configs/checkpoints, MIT license, and the repository's pinned legacy MMDetection 2.25.3/MMCV 1.5.0 stack. |
| E1f | https://github.com/OpenGVLab/InternImage | Official InternImage-T ImageNet-1K backbone and COCO Mask R-CNN detection releases, MIT license. |
| E1g | https://github.com/SwinTransformer/Swin-Transformer-Object-Detection | Official Swin-T ImageNet-1K backbone and COCO Mask R-CNN detection releases. |
| E1h | https://github.com/FocalNet/FocalNet-DINO | Official FocalNet DINO object-detection integration and public checkpoints. |
| E1i | https://github.com/facebookresearch/ConvNeXt-V2 | Official FCMAE ImageNet-1K weights. Code is MIT; official ImageNet weights are CC-BY-NC. This remains a pretraining-transfer candidate until attached to a verified detector. |
| E1j | https://github.com/facebookresearch/dinov2 | Official ViT-S/14 LVD-142M weights and Apache-2.0 standard DINOv2 model card. This remains a pretraining-transfer candidate until attached to a verified detector. |

The smoke CLI writes `passed`, `incompatible`, or `implementation_failure`. It never records AP, ranking, or promotion evidence. The local environment currently lacks MMDetection and the official D-FINE source checkout, so CR1, DF1, and DF2 cannot pass locally until those exact dependencies and checkpoints are supplied. E1d through E1h require their official repositories and version-pinned compiled dependencies. E1i and E1j can smoke their 16-band backbones locally, but cannot pass E1-0 until a real detection neck and head produce valid boxes.
