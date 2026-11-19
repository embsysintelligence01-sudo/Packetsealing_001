# Packet Mouth Lab — packet bounding-box classification

This experiment uses the existing packet ONNX detector to provide the entire packet bounding box for dataset collection and classification. No manually drawn mouth ROI is required.

## Run

From the project root:

```bash
./run_packet_sealing.sh --mouth-lab
```

Open a supplied photo directly:

```bash
./run_packet_sealing.sh --mouth-lab --source packet_mouth_lab/sample_images/2026-11-10-155451.jpg
```

Use **Open image**, **Open video**, or **Open camera**. The existing `bagnewmodel.onnx` and `fp_filter_resnet18_bag.onnx` locate the packet using production thresholds (0.5 segmentation, 0.6 filter). A blue box and tinted mask display the detection. The detector does not determine the opening state itself.

USB camera capture requests the saved `usb.resolution` from the main tracker's `camera.json` (currently **2560×1440**), with 2560×1440 as the fallback. `JEWEL_TRACKER_DIR` selects the tracker directory when configured. The lab requests MJPG before setting the dimensions and displays both requested and actual frame size, since cameras can deliver a different supported resolution. Processing remains capped at 960 pixels wide, matching the sealing app. Image and video files retain their source resolution before that processing resize.

## Collect, train, display results

1. Leave **Enable trained packet classification** unticked to collect without predictions. You can tick it at any time; if no packet-box classifier exists, the app explains that training is required and does not invent predictions.
2. Show exactly one detected packet. Choose `OPEN_CAMERA`, `OPEN_SIDE`, `CLOSED_FOLDED`, or `UNCLEAR`, enter the physical packet / recording group ID, then click **Save manually labelled example**. The crop comes directly from the detected packet box.
3. Keep all frames from one recording in the same group. Collect multiple independent recordings/packets per class, including folds, hands, reflections and printed borders. Label ambiguous views `UNCLEAR`. A sideways packet is only `OPEN_SIDE` when the opening is visibly open.
4. Pause playback and select **Train + held-out evaluation (CPU)**. Training requires at least three classes (including closed/folded and an open class), at least 12 examples in each represented class, and enough independent groups to hold out every class. Initially aim for 50–100 diverse examples per class from 3–5 independent groups; more may be needed.
5. Training enables and ticks the classifier control. Results such as `OPEN_CAMERA 91%` or `OPEN_SIDE 86%` appear on the packet box and in the sidebar. Uncheck it to resume collection without predictions. The selected manual label remains under your control; predictions never automatically label the dataset.
6. Replay unseen recordings to assess errors. Frame predictions are distinct from temporal consistency; paused images do not establish temporal confirmation.

No detections or multiple packets prevent classification and collection. Previous boxes/predictions are not reused. The whole packet box includes the visible contents, printing and nearby background inside the rectangle; it is not a mouth localisation result.

## Saved data

New full-packet training samples live in `dataset/packet_bbox/`. The earlier mouth-ROI dataset is preserved separately and is not mixed into training.

- Each labelled sample has a packet-crop JPEG and JSON containing the class, recording group, `input_region: packet_bbox`, and a link to its context capture.
- Context captures in `dataset/packet_bbox/packet_frames/<capture>/` contain `frame.jpg`, `packet.jpg`, `packet_mask.png` and `metadata.json`. Metadata includes source path/type/time, packet box and score, frame size, and the manual `packet_label`. Mouth ROI is null because this workflow does not annotate one.
- **Save detected packet frame (unlabelled)** captures the same context without assigning a class.
- The classifier is `models/packet_bbox_classifier.joblib`; its evaluation is `models/packet_bbox_classifier.validation.json`. Older `mouth_classifier.joblib` files are not used.
- **Export current evidence frame** saves the displayed packet result, packet crop and prediction metadata in `evidence/`.

Coordinates refer to the working frame, resized to at most 960 pixels wide. Scores from the bag detector and classifier have different meanings; they do not certify physical opening, insertion, or sealing.

## Environment and verification

The launcher uses the existing SLIC environment, which already contains the required detector dependencies on this machine. For a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r packet_mouth_lab/requirements-packet.txt
python -m pip install -r packet_mouth_lab/requirements-training.txt
python -m packet_mouth_lab.app
```

`--smoke-ui` constructs the window and exits. With `--source`, it also opens and detects the supplied image or initial video frame. OpenCV's injected Qt plugin paths are cleared for the PyQt display. Do not run the lab and main application against the same camera simultaneously.

```bash
../jewel_tracker/jewel_tracker/.venv-slic/bin/python -m unittest discover -s packet_mouth_lab/tests -v
```

This is an experimental logistic-regression classifier using HOG, colour and edge features. Whole-packet classification can learn background/printing correlations and confuse folds with openings. Validation on independent recordings is required before production use. The main sealing acceptance rule is unchanged.
