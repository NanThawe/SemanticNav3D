# SemanticNav3D

**CPU-efficient RGB-D semantic spatial mapping for robot perception**

SemanticNav3D combines visual SLAM, semantic keyframe selection, object detection, lightweight 3D localisation, persistent object mapping, and object-level scene graphs in a CPU-oriented perception pipeline.

The project focuses on object-level spatial understanding rather than dense 3D reconstruction. Each persistent object is represented by a robust 3D position in the SLAM world frame and can be queried semantically or used for camera-relative target guidance.

## Pipeline

```text
RGB-D sequence
    |
    v
ORB-SLAM3
    |
    v
Camera trajectory
    |
    v
Semantic keyframe selection
    |
    v
YOLO11n object detection
    |
    v
RGB-D object localisation
    |
    v
Persistent object association
    |
    v
Semantic object map
    |
    v
Object-level scene graph
    |
    v
Semantic queries and target guidance
```

## Main Features

- RGB-D visual localisation with ORB-SLAM3
- Semantic processing only on selected SLAM keyframes
- Pretrained YOLO11n object detection on CPU
- Robust object-level 3D localisation from RGB-D depth
- Persistent semantic object tracking across frames
- Object-level spatial scene graph generation
- Camera-relative semantic target queries
- Lightweight directional guidance
- CPU timing and cross-sequence evaluation

## Why Semantic Keyframes?

Running object detection on every RGB-D frame is unnecessary for a persistent semantic map.

SemanticNav3D uses SLAM keyframes as the semantic processing schedule. Across the two evaluated TUM RGB-D sequences, this avoided approximately **73 to 75% of YOLO invocations**.

## Evaluation

The same detector, association, persistence, and scene-graph parameters were used on both sequences.

| Metric | Freiburg1 Desk | Freiburg1 Room |
|---|---:|---:|
| RGB-D frames | 596 | 1,362 |
| ORB-SLAM3 ATE RMSE | 1.68 cm | 6.90 cm |
| Semantic keyframes | 147 | 371 |
| Semantic workload | 24.66% | 27.24% |
| Detector invocations avoided | 75.34% | 72.76% |
| YOLO detections | 598 | 1,034 |
| Valid 3D localisations | 574 | 1,006 |
| 3D localisation success | 95.99% | 97.29% |
| Persistent objects | 32 | 68 |
| Mean object-position RMSE | 7.9 cm | 8.1 cm |
| Median object-position RMSE | 6.4 cm | 7.6 cm |
| Mean semantic track purity | 0.978 | 0.978 |
| Scene-graph nodes | 32 | 68 |
| Scene-graph edges | 68 | 104 |
| Mean YOLO CPU latency | 30.64 ms | 31.33 ms |
| Mean semantic keyframe update | 42.80 ms | 44.38 ms |
| Amortised semantic cost per input frame | 10.56 ms | 12.09 ms |

A notable result is that the room sequence had substantially higher SLAM trajectory error, while persistent object-map consistency remained similar. Mean object-position RMSE changed from **7.9 cm to 8.1 cm**, and mean semantic track purity remained **0.978**.

## CPU Efficiency

For `freiburg1_room`:

```text
Total RGB-D frames:              1362
Semantic keyframes:               371
Detector frames skipped:          991
Detector invocations avoided:   72.76%

YOLO on every frame:           ~42.68 s CPU
YOLO on semantic keyframes:    ~11.63 s CPU
Estimated detector time saved: ~31.05 s

Amortised semantic cost:        12.09 ms/input frame
```

The system is designed to reduce semantic compute rather than maximise dense scene representation.

## Object-Level 3D Localisation

For each 2D detection:

1. Depth is sampled from the central region of the bounding box.
2. Invalid depth values are removed.
3. Median depth and MAD-based filtering provide a robust representative depth.
4. The bounding-box centre is projected into 3D using the RGB-D camera intrinsics.
5. The point is transformed from camera coordinates into the SLAM world frame.

The result is one lightweight representative 3D position per object observation.

## Persistent Object Mapping

Repeated observations are associated using:

- semantic class compatibility
- 3D spatial distance
- recent object position
- temporal proximity
- bounding-box overlap for nearby observations

Tracks must meet a minimum observation count before being retained as persistent objects.

## Scene Graph

Persistent objects become scene-graph nodes.

The current graph uses two conservative distance-based relations:

- `VERY_NEAR`
- `NEAR`

The graph can be used for neighbourhood queries, semantic object lookup, and target-relative reasoning.

## Semantic Guidance

A mapped semantic target can be transformed into the current camera frame to produce simple directional guidance:

```text
TURN_LEFT
TURN_RIGHT
FORWARD
TURN_AROUND
ARRIVED
```

This is **camera-relative directional guidance**, not full autonomous navigation.

The current system does not perform:

- path planning
- traversable free-space modelling
- collision avoidance
- robot control

## Repository Structure

```text
SemanticNav3D/
├── README.md
├── requirements.txt
├── src/
│   ├── frame_source.py
│   ├── tum_rgbd.py
│   ├── pose.py
│   ├── tum_groundtruth.py
│   ├── trajectory_eval.py
│   ├── object_detector.py
│   ├── object_localization.py
│   ├── object_map.py
│   ├── scene_graph.py
│   ├── scene_graph_query.py
│   ├── robot_relative_query.py
│   └── semantic_guidance.py
├── scripts/
│   ├── create_tum_associations.py
│   ├── evaluate_slam.py
│   ├── select_keyframes.py
│   ├── detect_keyframes.py
│   ├── localize_objects_3d.py
│   ├── build_object_map.py
│   ├── analyze_object_map.py
│   ├── build_scene_graph.py
│   ├── benchmark_semantic_pipeline.py
│   └── demo_tum_semanticnav.py
└── results/
    ├── desk/
    └── room/
```

Datasets, YOLO weights, ORB-SLAM3 build files, virtual environments, and large generated outputs are intentionally excluded from the repository.

## Setup

### Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### ORB-SLAM3

ORB-SLAM3 is an external dependency and is not bundled with this repository.

The RGB-D pipeline expects ORB-SLAM3 to produce:

```text
CameraTrajectory.txt
KeyFrameTrajectory.txt
```

These trajectories are then consumed by the Python semantic pipeline.

### Dataset

The current evaluation uses TUM RGB-D sequences.

A local layout can look like:

```text
data/
└── tum/
    └── rgbd_dataset_freiburg1_room/
```

Raw datasets are not included in the repository.

## Example Workflow

Create the TUM RGB-D association file:

```powershell
python scripts/create_tum_associations.py `
  --sequence data\tum\rgbd_dataset_freiburg1_room
```

Evaluate the ORB-SLAM3 trajectory:

```powershell
python scripts/evaluate_slam.py `
  --sequence data\tum\rgbd_dataset_freiburg1_room `
  --estimated results\slam\freiburg1_room\CameraTrajectory.txt `
  --output results\slam\freiburg1_room\evaluation
```

Select semantic keyframes:

```powershell
python scripts/select_keyframes.py `
  --sequence data\tum\rgbd_dataset_freiburg1_room `
  --keyframes results\slam\freiburg1_room\KeyFrameTrajectory.txt `
  --output results\keyframes\freiburg1_room
```

Run YOLO11n on the semantic keyframes:

```powershell
python scripts/detect_keyframes.py `
  --sequence data\tum\rgbd_dataset_freiburg1_room `
  --keyframes results\keyframes\freiburg1_room\keyframes.csv `
  --output results\detections\freiburg1_room `
  --model yolo11n.pt `
  --confidence 0.35 `
  --image-size 640
```

Estimate object-level 3D positions:

```powershell
python scripts/localize_objects_3d.py `
  --sequence data\tum\rgbd_dataset_freiburg1_room `
  --detections results\detections\freiburg1_room\detections.csv `
  --trajectory results\slam\freiburg1_room\CameraTrajectory.txt `
  --output results\objects3d\freiburg1_room.csv
```

Build the persistent object map:

```powershell
python scripts/build_object_map.py `
  --objects3d results\objects3d\freiburg1_room.csv `
  --output results\object_map\freiburg1_room `
  --association-distance 0.25 `
  --min-observations 3 `
  --final-merge-distance 0.05
```

Build the scene graph:

```powershell
python scripts/build_scene_graph.py `
  --objects results\object_map\freiburg1_room\object_map.csv `
  --output results\scene_graph\freiburg1_room `
  --very-near-distance 0.25 `
  --near-distance 0.40
```

Run the CPU semantic benchmark:

```powershell
python scripts/benchmark_semantic_pipeline.py `
  --sequence data\tum\rgbd_dataset_freiburg1_room `
  --trajectory results\slam\freiburg1_room\CameraTrajectory.txt `
  --keyframes results\keyframes\freiburg1_room\keyframes.csv `
  --model yolo11n.pt `
  --confidence 0.35 `
  --image-size 640 `
  --association-distance 0.25 `
  --min-observations 3 `
  --output results\benchmark\freiburg1_room
```

For complete options:

```powershell
python scripts/<script_name>.py --help
```

## Metric Notes

The **3D localisation success rate** measures whether a detection had usable depth and an associated camera pose. It is not ground-truth positional accuracy.

The **object-position RMSE** measures consistency among observations assigned to the same persistent track. It is not absolute ground-truth object localisation error.

Persistent object counts represent retained tracks, not annotated physical-object counts.

Benchmark mapping counts can differ slightly from the offline mapping pipeline because the benchmark uses an incremental localisation and association path.

## Current Limitations

- Pretrained COCO labels can be imperfect for indoor objects.
- Dynamic classes such as `person` can produce unstable or fragmented tracks.
- Object association is geometric and heuristic rather than learned.
- Each object is represented by a single 3D point rather than full 3D extent.
- The scene graph currently models proximity only.
- ORB-SLAM3 is run externally rather than through a fully integrated Python interface.
- Evaluation currently covers a limited number of indoor RGB-D sequences.
- Guidance does not include free-space planning or obstacle avoidance.

## Possible Extensions

- OpenVINO / ONNX CPU optimisation
- INT8 detector inference
- dynamic-object filtering
- learned object re-identification
- object orientation and 3D extent estimation
- additional scene-graph relations
- free-space and collision-aware planning
- live RGB-D camera integration
- evaluation on additional indoor RGB-D datasets

## Project Scope

SemanticNav3D is best described as:

> A CPU-efficient object-level semantic mapping system combining RGB-D visual SLAM, keyframe-based object detection, lightweight 3D localisation, persistent semantic tracking, scene-graph construction, and camera-relative target guidance.

It is not intended to be presented as:

- dense 3D reconstruction
- full autonomous navigation
- collision-aware path planning
- ground-truth object mapping
- a custom-trained object detector

## Technologies

Python, ORB-SLAM3, YOLO11n, OpenCV, NumPy, RGB-D perception, visual SLAM, 3D geometry, object association, scene graphs.

## Author

**Hom Nan Thawe Htun**

Master of AI, Nanyang Technological University

Interests: Computer Vision, 3D Vision, Robot Perception, Spatial AI
