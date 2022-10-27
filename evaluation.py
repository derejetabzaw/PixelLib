import pixellib
from pixellib.custom_train import instance_custom_training


train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 4)
train_maskrcnn.load_dataset("../pixellib")
train_maskrcnn.evaluate_model("mask_rcnn_models/mask_rcnn_model.001-2.103609.h5")