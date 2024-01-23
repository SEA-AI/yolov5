HOMEDIR=/home/eyesea/Git/yolov5/
MODEL=yolov5n6
echo "Training $HOMEDIR"

PYTHON_PATH=${HOMEDIR}/miniconda3/envs/AI/bin/python

python ${HOMEDIR}/train.py \
    --img 1280 \
    --batch 16 \
    --epochs 3 \
    --name ${MODEL}_RGB_D2304-v0_9C \
    --hyp ${HOMEDIR}/data/hyps/hyp.sea-ai.yaml \
    --device 0 \
    --data ${HOMEDIR}/datasets/TRAIN_THERMAL_DATASET_2023_06_2024-01-23/dataset.yaml \
    --weights ${HOMEDIR}/yolov5n6_RGB_D2304-v1_9C.pt \
# #    --workers 4 \
# #    --single-cls