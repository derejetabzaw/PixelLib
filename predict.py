import pixellib
from pixellib.instance import custom_segmentation

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 4, class_names= ["BG","API", "Area", "Line" , "Reel"])
segment_image.load_model("mask_rcnn_models/mask_rcnn_model.001-2.074585.h5")
segment_image.segmentImage("test/4.jpg", show_bboxes=True, output_image_name="test/4_out.jpg")