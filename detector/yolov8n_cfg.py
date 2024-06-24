from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolov8n/models/yolov8n.yaml'
cfg.WEIGHTS = 'detector/yolov8n/weights/yolov8n.pt'
cfg.INP_DIM =  640
cfg.NMS_THRES =  0.6
cfg.CONFIDENCE = 0.1
cfg.NUM_CLASSES = 80
