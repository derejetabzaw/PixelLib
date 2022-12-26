import pixellib
from pixellib.custom_train import instance_custom_training
import sys
print(sys.version)

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 5)
train_maskrcnn.load_dataset("../PIXELLIB")
train_maskrcnn.evaluate_model("mask_rcnn_models/mr_e10_110322.h5.h5")