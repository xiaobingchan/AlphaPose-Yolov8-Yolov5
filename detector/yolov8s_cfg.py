from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolov8s/models/yolov8s.yaml'
cfg.WEIGHTS = 'detector/yolov8s/weights/yolov8s.pt'
cfg.INP_DIM =  640
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.1
cfg.NUM_CLASSES = 80
