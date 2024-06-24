原始Docker构建方法：


docker pull nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
export http_proxy=http://192.168.2.33:17890
export https_proxy=http://192.168.2.33:17890
git clone https://github.com/gmt710/AlphaPose_yolovx
docker run --rm -it --gpus all -v /home/luyanjie3/AlphaPose_yolovx:/AlphaPose_yolovx luyanjie/cuda:10.1-cudnn7-devel-ubuntu18.0-alphapose-yolov5 /bin/bash

https://blog.csdn.net/qq_35975447/article/details/114940943

apt-get update -y
apt-get install python3 python3-pip python3.6-tk wget libgtk2.0-dev pkg-config -y
apt-get install -y gcc zlib1g-dev libjpeg-dev git
cd /AlphaPose_yolov8
export http_proxy=http://192.168.2.33:17890
export https_proxy=http://192.168.2.33:17890
pip3 install --upgrade pip
pip3 install requests pandas seaborn
pip3 install numpy Cython pyyaml scipy==1.5.4 matplotlib munkres googledrivedownloader scikit-image Pillow websocket-client wf-pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install opencv-python==4.7.0.72 numpy==1.19.5 easydict==1.10 natsort==8.2.0 pycocotools==2.0.6 Cython -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
wget https://download.pytorch.org/whl/cu101/torch-1.7.1%2Bcu101-cp36-cp36m-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu101/torchvision-0.8.2%2Bcu101-cp36-cp36m-linux_x86_64.whl
wget https://download.pytorch.org/whl/torchaudio-0.7.2-cp36-cp36m-linux_x86_64.whl

pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python3 setup.py build develop --user
python3 -m pip install git+https://gitee.com/DwRolin/cython_bbox.git

已经打包好了镜像上传到dockerhub：
docker pull luyanjie/cuda:10.1-cudnn7-devel-ubuntu18.0-alphapose-yolov5

使用方法：
git clone https://github.com/luyanjie3/AlphaPose_yolov8
docker run --rm -it --gpus all -v /home/luyanjie3/AlphaPose_yolov8:/AlphaPose_yolov8 luyanjie/cuda:10.1-cudnn7-devel-ubuntu18.0-alphapose-yolov5 /bin/bash

cd /AlphaPose_yolov8
python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml --checkpoint pretrained_models/fast_dcn_res50_256x192.pth --indir examples/demo/ --showbox --save_img --pose_track --sp --vis_fast --detector yolov5

python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml --checkpoint pretrained_models/fast_dcn_res50_256x192.pth --indir examples/demo/ --showbox --save_img --pose_track --sp --vis_fast --detector yolov8n

python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml --checkpoint pretrained_models/fast_dcn_res50_256x192.pth --indir examples/demo/ --showbox --save_img --pose_track --sp --vis_fast --detector yolov8s

检测的图片和视频放在：examples/demo/
生成的结果在：examples/res/vis/