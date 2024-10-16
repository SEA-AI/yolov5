HOMEDIR=/home/kevinserrano
MODEL=yolov5n
${HOMEDIR}/miniconda3/envs/AI/bin/python ${HOMEDIR}/GitHub/yolov5/train.py \
    --img 640 \
    --batch 32 \
    --epochs 100 \
    --data ${HOMEDIR}/GitHub/yolov5/datasets/TRAIN_THERMAL_DATASET_2023_06_2023-08-01/dataset.yaml \
    --weights yolos/yolov5n.pt \
    --name ${MODEL}_T16-8_D2306-v0_9C \
    --hyp ${HOMEDIR}/GitHub/yolov5/data/hyps/hyp.sea-ai-IR.yaml \
    --workers 4 \
    --device 0
#    --single-cls
