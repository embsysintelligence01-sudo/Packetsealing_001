# Automatic packet sealing

```bash
./run_packet_sealing.sh
```

The application connects to the saved camera and starts inference automatically.
There are no Capture, Mark Opening, Recheck, or Hand Check / Seal process buttons.
Camera/bench ROI controls are setup controls; Finish / Save ends an incomplete
session and exports its evidence.

**Reset / Start Over** saves the current session, waits for recording/export to
finish, then automatically opens a new session using the same camera and bench
ROI. Jewel IDs, retained placements, packet ROI, strip history and sealing checks
start fresh. Completed-session evidence is preserved. Closing the window cancels
a pending automatic restart.

## Automatic flow

1. Lay all jewels separately in the bench ROI before starting. J1, J2, … assign
   automatically through Jewel Tracker's original stable-count routines, in
   top-to-bottom, left-to-right order. Gold outlines appear before assignment.
   SLIC/ResNet identity preparation runs in the background.
2. The bag segmentation model and its false-positive filter detect the **full
   bag mask**, which becomes the packet ROI automatically. No opening polygon
   is used. Legacy saved opening coordinates are ignored.
3. Picking up a known jewel starts **Taken / interacting** immediately, even
   before bag detection or gallery preparation finishes. The original
   `visibility()`, `choose_hand_jewel()` and `PICKUP_STATES` trigger is retained.
   A segmented hand must also approach an initial assigned bench spot: this
   prevents HSV gold/skin confusion and ID swaps during transfer. Gold must
   leave that bench spot on two fresh observations, so passing a hand over a
   bangle still on the bench cannot trigger the next cycle. The initial
   spot determines ownership; ambiguous pickups record without guessing an ID.
4. The pickup owns that jewel's ID throughout transfer. Placement uses an OR:
   its associated courier hand reaches the packet, or moving gold from that
   pickup's bench-to-packet path reaches the packet. Gold is matched by motion
   continuity, without SLIC/ResNet reidentification inside the bag. Each gold
   candidate belongs to one path per frame. Already-visible packet contents,
   other bench jewels, ambiguous candidates and stale gold results cannot
   confirm a new transfer. Gold inference continues for the final taken jewel
   until its transfer completes. The hand path includes a small packet boundary
   band; the gold path checks segmented overlap or an in-mask point. Packet position may change freely during
   transfer. Between model detections, optical flow can follow the moving packet
   for up to 2.5 seconds, with feature-consistency checks. This tracked region is
   shown in amber and is used only for transfer interaction, never strip or
   sealing verification. When tracking is unavailable, the interaction stage
   stays retained without inventing a placement. The other hand holding the bag cannot
   initiate placement from across the bench. No SLIC/ResNet classification of
   the carried or hidden jewel is required.
5. Arrival at the packet records **Placed in packet — inferred** immediately.
   No disappearance, stationary-gold, hands-clear or stationary-bag wait is
   required. A jewel may remain visible through the opening or plastic.
   This result stays latched through later occlusion, packet movement and the
   next pickup. Hand-tracking loss without an observed packet interaction
   retains **Taken**, rather than inventing an inside result.
6. The next pickup closes the previous jewel's clip at that same source
   timestamp and starts the next jewel's clip with one second of available
   pre-roll. There is no confirmation delay between jewels. A previous pickup
   with no observed packet interaction stays **Taken / interaction retained**.
   A new pickup never turns the outgoing jewel into placement unverified or
   resets its transfer. The handoff frame checks the outgoing transfer before
   starting the next courier, without classifying either jewel inside the bag. The last
   jewel's clip continues until Finish/Save or session completion. Reset / Start
   Over saves the session before beginning again with fresh IDs.
7. After all placements, put the packet on the test bed **with its strip
   visible**. Three strip observations and a stationary packet establish the
   test-bed baseline once. The app then prompts for strip removal.
   This is a one-way stage: packet movement, a bag-detector dropout or a later
   strip false positive never restarts baseline verification or a completed
   removal. Packet stationarity measures geometry independently of hand presence.
8. Strip removal needs five fresh absent-strip observations spanning at least
   one second, with the strip area in view and not covered by a hand. Only hands
   near the packet count; a false hand detection on a distant fixture does not
   block this step. If the full-bag detector fails, a visual feature reference
   of the same packet body supplies the inspection region. The entire bag does
   not need to remain visible. An empty bed or an off-camera strip area cannot
   prove removal. Rejected HSV strip candidates remain uncertain.
9. Once strip removal is confirmed, mark **Sealed — inferred from strip removal**
   immediately. No separate closure model or further sealing wait is required.
   The result stays retained through later packet movement or missing detections.
10. The camera and recording keep running. The operator is prompted to rotate
    the packet and show its number inside the bench ROI. `pcnumber.onnx` detects
    the rotated number region, which is rectified and read by PaddleOCR. Three
    matching reads, each with at least 85% confidence, spanning at least one
    second confirm the number and automatically save the session. Rotation itself
    is not verified. Missing, clipped or multiple number regions cannot confirm.
    If OCR fails or the operator finishes early, the sealed result is retained
    and the report explicitly records that the packet number was not confirmed.

An opaque packet's contents are **inferred**, not directly inspected. This method
also cannot distinguish a jewel hidden underneath the bag from one inside it
with certainty. A courier reaching the opening is evidence of a packing action, not proof
that the jewel was released or remained inside. Physical adhesive strength is not measured.

## Sealing rule

The application uses the verified strip-removal sequence as its sealing rule.
The strip must first be seen on the stationary packet at the test bed. Then five
fresh absent-strip observations spanning at least one second, with packet/body
context, the strip area visible and packet hands clear, confirm removal.
A detector miss, uncertain strip prediction or covered strip cannot finish it.
This is process inference; pressing the adhesive flap shut is not directly
measured. No closure weights or `PACKET_CLOSURE_CONFIG` are needed by the app.

## Isolated PaddleOCR setup

```bash
bash setup_packet_ocr.sh
./run_packet_sealing.sh --check-ocr
```

PaddleOCR 3.3.3 and PaddlePaddle 3.2.2 run in `.venv-ocr` (Python 3.10).
The first self-check downloads the English mobile detection/recognition and
text-line orientation weights. Later runs use the local cache. The SLIC
interpreter runs the number ONNX detector; a persistent subprocess runs PaddleOCR
with two CPU threads. Only one crop request is in flight, so camera preview and
recording continue independently. Set `PACKET_OCR_PYTHON` to use another compatible
OCR interpreter. Missing dependencies, timeouts and OCR errors remain visible;
Finish / Save saves the available evidence without inventing a packet number.

The single PDF includes the confirmed number, confidence, capture time and crop.
`packet_number_crop.jpg` and `packet_number_scene.jpg` are saved alongside the
PDF. The shared `packet_sealing.avi` includes the rotation/number capture; the last
individual jewel clip ends at sealing so it does not include the OCR wait.

## Tracking and camera

The app uses the existing sibling `jewel_tracker/jewel_tracker/.venv-slic` and
version 5's INT8 gold detector, position/Hungarian assignment, SLIC signatures,
ResNet identity gallery, and pickup visibility policy. Gold inference loads
`int_8_GOLD_JEWEL/gold_int8.xml` and its `.bin` from the tracker directory.
Original tracker, model, environment, Flask route and service files are unchanged.

The saved bench ROI comes from the tracker's `zones.json`. USB resolution and
anti-flicker settings come from `camera.json`, preserving coordinate alignment.
Use Jewel Tracker ROI reloads that geometry; drawing a bench ROI makes a local
packet override. Bench configuration freezes at the first pickup. Start a new
session to change it afterward. Run one application per camera.

Live camera display uses an independent 40 ms timer. Gold overlays publish
before packet inference completes; expired overlays are not displayed as live.
The preview renders at up to 1280 pixels wide with mouse coordinates preserved
in full camera pixels. Masks are outlines, without a pink hand tint. Packet
masks decode at up to 960 pixels wide and map back to camera coordinates.

Set `JEWEL_TRACKER_DIR` if the tracker folder moves. `PACKET_PYTHON` can override
its interpreter but must supply the existing SLIC/OpenCV contrib dependencies.

## Models and evidence

All five files in `model onnx/` are required: `bagnewmodel.onnx`, `stripmodel.onnx`,
`handseg model.onnx`, `fp_filter_resnet18_bag.onnx`, `hsv_fp_filter_strip.pt`.
Packet exports currently use floating-point ONNX; gold uses OpenVINO INT8.
Bag FP classification follows the original RGB/ImageNet 224×224 normalization
and true-positive class 1. The export has no label metadata; validate that mapping
on known positive and negative examples. Strip FP classification uses the
original 64×64 HSV CNN. A filter failure interrupts verification instead of
silently bypassing it.

Recording is independent of inference at 10 FPS, up to 960 pixels wide, with a
four-second rolling buffer. Pickup enables recording. Individual exported clips
start one second before pickup when available, matching the original tracker
clip-start rule. `outputs/<session>/` contains:

- `report.pdf`: one visual session report for all jewels, with a session overview,
  Last updated time, per-jewel photos, pickup/placement/handoff times and video
  links. Two jewels produce four pages: one overview with jewel cards and
  packet number/OCR crop, then one page per jewel with its original thumbnail,
  recording link and three timestamped images (pickup, packet interaction, retained
  placement), followed by a sealed-packet page with the sealing-time image, start/end
  timestamps and duration. Sealing starts after all placements and ends at confirmed
  strip removal, before OCR. Every page carries the packet number. It uses Jewel Tracker's navy/teal audit style. No individual jewel PDFs
  or additional summary PDF are created for new sessions.
- `J1.avi`, `J2.avi`, etc.: individual transfer clips linked from the shared PDF.
- `packet_sealing.avi`: sealing/incomplete-session video. Ambiguous pickup
  recordings remain unassigned rather than guessed.
- `evidence.json`: result and media validation metadata, not a routine event log.

Reports include the placement method (`pickup_owned_hand_packet_interaction`
or `pickup_owned_gold_packet_interaction`), inferred sealing method/status, and recording coverage
issues. A recording issue flags review and blocks a final sealed report while
retaining the latched placement result; it does not reopen a confirmed jewel.

The target is 2,000,000 bytes per jewel clip, with a separate 2 MB budget
for the shared session PDF + sealing video. The shared PDF is charged once. FFmpeg/MJPEG AVI encoding preserves
complete intervals and decodes the output to verify duration. AVI uses Motion JPEG
(MJPG) for desktop player compatibility. These files can be larger than H.264 MP4;
the 2 MB target is retained, but complete evidence takes priority. Long/complex clips
can exceed that target at the minimum bitrate; the app reports this and retains
source evidence instead of truncating the video. Export failures preserve
`.working/`. No routine log panel or application log file is created. RTSP
credentials are not saved in packet UI settings.

## Checks

```bash
./run_packet_sealing.sh --check-models
./run_packet_sealing.sh --check-ocr
QT_QPA_PLATFORM=offscreen ./run_packet_sealing.sh --smoke-ui
../jewel_tracker/jewel_tracker/.venv-slic/bin/python -m unittest discover -s tests -v
```

Smoke/test modes do not open a physical camera. Live insertion/occlusion trials
are still needed; passing model-loading and synthetic tests does not measure
hardware accuracy.

## Check the supplied insertion video

Run deterministic replay in the SLIC environment:

```bash
./run_packet_sealing.sh --replay WIN_20261104_17_38_32_Pro.mp4
```

This runs the real INT8 gold, bag, strip, hand and false-positive models on
every sixth frame, using video timestamps and scaling the saved Jewel Tracker
bench ROI from the configured camera resolution. It writes
`outputs/video_replay.json` with pickup, placement and recording-handoff times.
Replay checks the transfer workflow; it does not assert sealing or export
recording evidence. Use `--replay-step 1` to inspect every frame.

Detection drawings persist across slower inference updates for up to 2.5 seconds.
After 0.75 seconds, they turn amber with a Last detection indicator. A missing
bag/strip contour can hold for up to 0.6 seconds from the last real detection;
repeated misses do not extend it. Display persistence never feeds pickup,
placement, strip removal or sealing decisions. Pickup immediately clears a
jewel's old bench marking.

Fast insertion handling uses the packet's full footprint, including the opening,
with a small boundary allowance. A single bag/flow miss may retain that region
for at most 0.45 seconds for transfer inference only. This prevents an incoming
bangle from being rejected merely because the detector later merges it with
already-packed gold. Local hand/gold paths can recover outside the packet after
a detection gap, without jumping onto the contents inside.
