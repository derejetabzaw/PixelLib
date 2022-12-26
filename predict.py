import pixellib
from pixellib.instance import custom_segmentation

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 5, class_names= ["BG","API", "Area", "Line" , "Reel","noresult"])
segment_image.load_model("mask_rcnn_models/mr_e10_110322.h5.h5")
segment_image.segmentImage("test/25.jpg", show_bboxes=True, output_image_name="test/25_out.jpg")