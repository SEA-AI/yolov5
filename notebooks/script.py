import sys
sys.path.append('..')

import fiftyone as fo
from fiftyone import ViewField as F
# delete any tmp dataset
[fo.delete_dataset(d) for d in fo.list_datasets() if d.startswith("tmp_")]

dataset_name = 'TRAIN_THERMAL_DATASET_2023_06'
try:
    dataset = fo.Dataset.from_dir(
        dataset_dir=f'/media/jigglypuff/NieuwVolume2/{dataset_name}',
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name,
        persistent=True,
    )
except ValueError as e:
    print(e)
    print("Loading existing dataset")
    dataset = fo.load_dataset(dataset_name)
    
if not dataset.has_field("uniqueness"):
    import fiftyone.brain as fob
    fob.compute_uniqueness(dataset)

noise_ratio = 0.25
annotated = dataset.exists('ground_truth_det.detections', True)
noise = dataset.exists('ground_truth_det.detections', False)

n_noise = min(len(noise), int(len(annotated) * noise_ratio))
subset : fo.DatasetView = annotated + noise.sort_by("uniqueness", reverse=True).take(n_noise, seed=42)
session.view = subset

val_ratio = 0.2
tags_suffix = 'v0'

# Split train/val based on trip
trip_counts = subset.count_values("trip")
# sort by count
trip_counts = dict(sorted(trip_counts.items(), key=lambda x: x[1], reverse=True))
# choose ramdom trips with approximate specified val_ratio
val_trips = [trip for trip in list(trip_counts)[::int(1/val_ratio)]]

train_n = sum([x[1] for x in trip_counts.items() if x[0] not in val_trips])
val_n = sum([x[1] for x in trip_counts.items() if x[0] in val_trips])
total_n = train_n + val_n
print(f"Ratio train:val split: {train_n/total_n:.2f}:{val_n/total_n:.2f}")

label_count_train = subset.match(~F("trip").is_in(val_trips)).count_values('ground_truth_det.detections.label')
label_count_val = subset.match(F("trip").is_in(val_trips)).count_values('ground_truth_det.detections.label')
plot_label_distribution(label_count_train, label_count_val)

subset.match(~F("trip").is_in(val_trips)).tag_samples(f"TRAIN_{tags_suffix}")
subset.match(F("trip").is_in(val_trips)).tag_samples(f"VAL_{tags_suffix}")

tmp_name = f'tmp_{dataset_name}'
try:
    export = subset.clone(tmp_name)
except ValueError:
    print(f"Loading existing dataset {tmp_name}")
    export = fo.load_dataset(tmp_name)
export: fo.DatasetView
print("# detections (before filtering):", 
      len(export.values(f"ground_truth_det.detections[].label")))

# read class map
import yaml
with open('../data/class_map.yaml', 'r') as f:
    class_map = yaml.load(f, Loader=yaml.FullLoader)

# apply category map
export = export.map_labels("ground_truth_det", class_map)
export = export.filter_labels("ground_truth_det", F("label") != "None", only_matches=False)
export.keep()
export.save()

print("# detections (after fltering):", 
      len(export.values(f"ground_truth_det.detections[].label")))

filepaths = export.values("filepath")
filepaths = [fp.replace("8Bit", "16Bit").replace("jpg", "png") for fp in filepaths]
export.set_values("filepath", filepaths)

from datetime import datetime
date = datetime.now().strftime("%Y-%m-%d")

export_dir=f"/home/eyesea/Git/yolov5/datasets/{dataset_name}_{date}"

classes = export.distinct("ground_truth_det.detections.label")
fo_splits = [f"TRAIN_{tags_suffix}", f"VAL_{tags_suffix}"]
yolo_splits = ["train", "val"]
for fo_split, yolo_split in zip(fo_splits, yolo_splits):
    print("split:", fo_split)
    split = export.match_tags(fo_split) # .take(5000, seed=51)
    split.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth_det",
        classes=classes,
        split=yolo_split,
    )

fo.delete_dataset(tmp_name)