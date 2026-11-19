Motion classification in `pickup_trigger_motion_class.py`

Use the existing Jewel Tracker Python environment. It supplies PyTorch,
torchvision, OpenVINO, the gold/stone detectors and the hand landmark runtime.
The main packing workflow is unchanged. Home-position mode in both pickup
experiments also offers a separate provisional hand-motion rubbing rule; select
Home-position logic for that grip–rubbing–return process. Its provisional outcomes
are not trained OK/NOT_OK predictions, and its recorded windows do not acquire
training labels automatically.

1. Collect several separate recordings:

```bash
../jewel_tracker/jewel_tracker/.venv-slic/bin/python pickup_trigger_motion_class.py --autostart
```

Select **Motion classification** (the default in this script). It detects the
stone on the full frame, uses the saved bench ROI, and saves each jewel's ordered
windows under `outputs/motion_classification/<recording>/motion_windows/`.
A window contains 16 observations by default (about 3 seconds at 5 FPS).
Use `--motion-window 32` for longer actions, consistently across recordings.

Each sample stores two RGB views: the bench context including stone and hand,
and the currently segmented jewel crop. Numeric cues include visibility,
normalized distance to the stone box, gold-mask overlap with the stone box,
stone-relative position/velocity, direction reversal, time near the stone,
sample interval and nearest fingertip distance. The detector box approximates
the stone ROI. Missing jewels have an explicit visibility flag and empty target
view; no inferred gold position is substituted. Identity switches in the upstream
tracker remain a limitation; merged detections are rejected.

Record correct pickup/approach/rubbing, immediate replacement, hovering, passing
near the stone, moving without contact, and stationary negatives. Include different
hands, jewels, lighting and speeds. Fully idle windows are collected too. Review
each window according to the action actually visible, not the intended action.

No checkpoint means **UNTRAINED**, never an automatic OK/NOT_OK label. Camera
instability, stone loss/movement and sampling gaps reset incomplete sequences.
Complete windows are non-overlapping; incomplete tails are discarded. Detector
models still receive full frames. Outside-ROI hands are filtered afterward.

2. Review and label complete windows:

```bash
../jewel_tracker/jewel_tracker/.venv-slic/bin/python pickup_trigger_motion_class.py --review-motion outputs/motion_classification
```

The preview loops each window. Press **O** for OK, **N** for NOT_OK, **S** to
skip uncertain/transition windows, **Q** to quit. Labels are saved in adjacent
JSON files; raw arrays remain unchanged. Already labeled windows are skipped.
You can clear a JSON `label` to `null` to review it again. OK should mean clearly
visible rubbing in this window, not merely proximity or a completed pickup.

3. Train after labeling both classes in at least two independent recordings:

```bash
../jewel_tracker/jewel_tracker/.venv-slic/bin/python pickup_trigger_motion_class.py --train-motion outputs/motion_classification --motion-checkpoint models/rubbing_motion.pt --epochs 15
```

Training uses frozen ImageNet ResNet-50 features from both views, concatenated
with numerical cues, then a trainable GRU and two-class head. Pretrained weights
may download on the first training run. Features are computed once before the
training epochs. The last recording session by path is held out for validation;
both training and validation must contain both classes. All resets within a run
share one session group. Replays of the same source across separate runs must
be kept in one group manually to avoid leakage. Validation selects the checkpoint;
use additional untouched recordings to estimate final performance.

4. Run trained inference:

```bash
../jewel_tracker/jewel_tracker/.venv-slic/bin/python pickup_trigger_motion_class.py --motion-checkpoint models/rubbing_motion.pt --autostart
```

Predictions appear in the status/log and `motion_predictions.jsonl`, and labeled
window collection remains enabled. Confidence below 0.8 becomes UNCERTAIN; windows
with less than 75% visible jewel observations become INSUFFICIENT_VISIBILITY.
These thresholds need validation; softmax confidence is not calibrated accuracy.
The fingertip movement event and learned action label are separate. OK means
visually consistent with rubbing, not verified physical contact or packet insertion.
ResNet inference runs on CPU at window completion and can add latency. No action
classifier has been trained or validated on real recordings in this change.
