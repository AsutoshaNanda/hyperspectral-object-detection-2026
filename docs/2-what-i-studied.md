# 2. What I studied and the resources I used

## My handwritten notes (18 Sep)

Six pages, in [notes/](../notes/).

| Page | What it covers |
|---|---|
| [1](../notes/page-1.jpg) | Basics: class, confidence, box. RGB has 3 bands, this data has 16. Plan 1 = YOLO26m with 16 bands (target 0.55 to 0.60), Plan 2 = RF-DETR-Small with 16 bands. Trust local validation and holdout over the public board. What mAP@[0.5:0.95] and IoU mean. Holdout = checking the model was not just lucky on validation |
| [2](../notes/page-2.jpg) | Validation as a mock exam. How to make validation strict. TTA (flipped and resized images, same model) is allowed; ensembles are not. Why COCO-pretrained weights help. Pseudo-RGB throws information away. X2Cube = height x width x 16 bands |
| [3](../notes/page-3.jpg) | Open questions from the discussions: is a DINO teacher / knowledge distillation allowed; what to do with visible but unlabelled objects. Idea from the ER-FlowScan report: extract a stable spectral profile of each object despite noise, shadow and light; self-attention across the 16 wavelengths; RF-DETR extends the 3-channel input layer to 16 channels |
| [4](../notes/page-4.jpg) | The hard pairs: egg vs egg_plastic, apple vs apple_plastic. Single-model ideas: better wavelength attention, spectral data augmentation, initialising the input layer for 16 bands, D-FINE's box-edge regression. Cascade R-CNN for tighter boxes, with multi-scale training |
| [5](../notes/page-5.jpg) | Methods to test: Cascade R-CNN, multi-scale training, pseudo-labeling (allowed?), Soft-NMS, cross-validation, patch training for small objects |
| [6](../notes/page-6.jpg) | Questions: why do small objects look bigger in patches; is normalization affecting results. Backbone, epoch, YOLO P2 level for small objects. Plan C = Plan B v2 but the 512 x 256 image is padded to 512 x 512 instead of stretched |

## Past competitions studied

| Competition | What I took from it | Link |
|---|---|---|
| PBVS 2022 hyperspectral object detection (closest past task) | Winner used Cascade R-CNN, multi-scale training, very confident pseudo-labels (above 0.99), multi-scale testing and Soft-NMS. Winning AP was 26.35 | [report](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Rangnekar_Semi-Supervised_Hyperspectral_Object_Detection_Challenge_Results_-_PBVS_2022_CVPRW_2022_paper.html), [1st place](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html) |
| Great Barrier Reef, 1st place | Trust cross-validation, not the public board | [writeup](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/qns-trust-cv-1st-place-solution) |
| Great Barrier Reef, 2nd place | Image resolution and multi-scale training mattered | [writeup](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/2nd-solution-yolov5) |
| Great Barrier Reef, 7th place | A single Cascade R-CNN can do well | [writeup](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/writeups/three-man-works-7th-place-solution-cascade-rcnn-tr) |
| Global Wheat Detection, 2nd place | Check suspicious boxes, validate before adding tricks | [writeup](https://www.kaggle.com/competitions/global-wheat-detection/writeups/overfeat-2nd-place-solution-with-code-mit-complian) |
| Global Wheat Detection, 9th place | Build folds balanced by box count, box size and source | [discussion](https://www.kaggle.com/competitions/global-wheat-detection/discussion/172569) |
| Airbus Ship Detection, 4th place | When the public board uses a small part of the data, do not chase small changes | [writeup](https://www.kaggle.com/competitions/airbus-ship-detection/writeups/attention-heads-few-lessons-learned-4th-place) |

## Same-competition public work

- ER-FlowScan RF-DETR report: https://er-flowscan.er-systems.org/optiscan-hsi/learn-architecture
  It reported pseudo-RGB validation 0.674 / public 0.58882 versus 16-band validation 0.6978 / public 0.62083. This was the main reason to keep all 16 bands. It is an outside result, not one I reproduced.

## Detectors and model code

| Model | Why it was considered | Link | What happened |
|---|---|---|---|
| YOLO26m (Ultralytics) | Simple, fast baseline that accepts 16 channels | [Ultralytics](https://github.com/ultralytics/ultralytics), [multispectral docs](https://docs.ultralytics.com/datasets/detect/coco8-multispectral) | Plan A |
| RT-DETR-L (Ultralytics) | Its trainer accepts the channel count | [trainer source](https://github.com/ultralytics/ultralytics/blob/v8.4.147/ultralytics/models/rtdetr/train.py) | Main model family |
| RF-DETR | Same-competition report above | [loader source](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/datasets/coco.py) | Not used: its standard loader converts images to RGB |
| D-FINE S / M | Designed for tighter boxes (FDR, GO-LSD) | [D-FINE](https://github.com/Peterande/D-FINE) | Ported to 16 bands, tied Combo |
| RT-DETRv2 | Successor of RT-DETR | [RT-DETR repo](https://github.com/lyuwenyu/RT-DETR), [paper](https://arxiv.org/abs/2407.17140) | Ported, tied Combo |
| Cascade R-CNN | PBVS 2022 winner's detector | [paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html), [MMDetection config](https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn) | Ran on 24 Sep, lost (0.652) |
| DINO, Co-DETR, InternImage, Swin, FocalNet, ConvNeXt V2, DINOv2 | Outside-model screen (E1d to E1j) | [DINO](https://github.com/IDEA-Research/DINO), [Co-DETR](https://github.com/Sense-X/Co-DETR), [InternImage](https://github.com/OpenGVLab/InternImage), [Swin detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), [FocalNet-DINO](https://github.com/FocalNet/FocalNet-DINO), [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2), [DINOv2](https://github.com/facebookresearch/dinov2) | Not run (see plan doc) |

## Techniques and their sources

| Technique | Link |
|---|---|
| Soft-NMS | https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html |
| Sliced inference for small objects (SAHI) | https://github.com/obss/sahi, https://doi.org/10.1109/ICIP46576.2022.9897990 |
| MNF denoising | https://pubs.usgs.gov/publication/ofr20251038/full, https://www.mathworks.com/help/images/ref/hypermnf.html |
| Spectral Angle Mapper (SAM) | https://www.mathworks.com/help/images/ref/sam.html, https://www.sciencedirect.com/science/article/pii/S1537511006004065 |
| Spectral normalization (SNV, area, z-score) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9541333/, https://pmc.ncbi.nlm.nih.gov/articles/PMC13003176/ |
| MSC, SNV, Savitzky-Golay | https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1211617/full |
| COCO AP evaluator (size AP) | https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py |
| Ultralytics train / predict settings | https://docs.ultralytics.com/modes/train, https://docs.ultralytics.com/modes/predict |

## Competition discussions I read

- [727863](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863) single model, TTA, pretrained weights
- [737136](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737136) are all visible objects labelled
- [741902](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/741902) label cleaning and pseudo-labeling
- [739853](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/739853) self-training on test images, WBF
- [742296](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742296) team information reminder
- [742487](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742487) Phase 2 notice
- [743224](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743224) Phase 1 results

## Other things checked

- A Roboflow basketball-tracking video (25 Sep). Almost all of it was about video tracking, so it did not apply to single 16-band images.
