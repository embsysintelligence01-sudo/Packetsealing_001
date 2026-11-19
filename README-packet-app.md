# Automatic packet sealing

```bash
./run_packet_sealing.sh
```

The main process also loads the locally trained
`packet_mouth_lab/models/packet_bbox_classifier.joblib` for advisory packet-view
classification. It uses the detected packet box, a maximum 960-pixel working
width, and the same 64×64 feature preprocessing as the lab. Live overlays show
the class, score, and whether it is temporally consistent. Only current detector
boxes are classified; motion-only and saved-body regions are not classifier input.

Consistency requires at least three timestamp-separated predictions spanning
0.4 seconds, each scoring at least 0.78, with no gap over 1.1 seconds. The score
threshold is a heuristic, not a guarantee of correctness. Low-score, missing,
or inconsistent observations cannot confirm an opening. The classifier neither
confirms jewel insertion nor changes the placement/sealing acceptance rules.

Each jewel's PDF page reports the classification from its exact placement frame,
including the class, score, timestamp, and whether temporal consistency was
established. A single-frame prediction is reported even without temporal
confirmation; it remains separate from placement inferred by the existing
transfer logic. A missing accepted packet detection at placement is reported as
unavailable, without substituting a later frame. Clean camera frames and packet
crops are saved before overlays as `J<id>_placement.jpg` and
`J<id>_placement_packet.jpg`, and preserved with the exported PDF. These images
use the maximum 960-pixel working width. The placement snapshot is retained
through subsequent detector dropouts and changes in view. Source-time
observations, the classification at placement, model training time and evaluation
are included in `evidence.json`. The report displays the model's saved held-out
accuracy. Missing or incompatible classifiers leave the main workflow running
with an unavailable/error status; collecting and retraining in the lab does not
require changes to the sealing rules. Restart the main application to load an
updated classifier.

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
4. The pickup owns that jewel's ID throughout transfer. Hand contact with the
   packet is recorded as **At packet**, with insertion unconfirmed. It cannot
   create a placement image or an inferred inside result by itself.
   Inferred transfer requires either a continuously associated hand observed
   outside the packet and then crossing its boundary in two fresh arrival
   observations, or pickup-owned moving gold observed outside and arriving at
   the packet. Packet movement under a stationary hand, large packet shape
   changes, ambiguous hand associations and blind gaps cannot supply hand-arrival
   proof. The supplied model detects the packet body, not its physical mouth;
   boundary crossing remains supporting evidence, not verified hidden insertion.
   Gold is matched by motion continuity without reidentification inside the bag;
   already-visible packet gold, other bench jewels and stale candidates are
   excluded. A stronger moving-gold result takes priority over hand contact.
   Between detector hits, feature-checked optical flow can follow the packet for
   up to 2.5 seconds for transfer context, never for strip-removal proof.
5. A supported arrival records **Placed in packet — inferred**. Plain contact
   stays unconfirmed even if the strip is detected or sealing finishes. A missed
   arrival is retained for review; it is not reconstructed from a later hand
   pressing or holding the packet. JSON exports include placement evidence and
   whether the physical opening was verified (currently false). Completed
   inferred results stay latched through subsequent occlusion or movement.
6. The next pickup closes the previous jewel's clip at that same source
   timestamp and starts the next jewel's clip with one second of available
   pre-roll. There is no confirmation delay between jewels. A previous pickup
   with no observed packet interaction stays **Taken / interaction retained**.
   A new pickup never turns the outgoing jewel into placement unverified or
   resets its transfer. The handoff frame checks the outgoing transfer before
   starting the next courier, without classifying either jewel inside the bag. The last
   jewel's clip continues until Finish/Save or session completion. Reset / Start
   Over saves the session before beginning again with fresh IDs.
7. The **first validated strip detection** starts sealing instructions and
   recording: “Remove the sealing strip.” This does not wait for recorded
   pickups, inferred jewel placements, three strip observations, or a stationary
   packet. The strip must pass the existing detector/filter and belong to an
   observed packet region. Unknown/rejected candidates cannot start this stage.
   Jewel statuses remain independent: unconfirmed pickups or placements are not
   converted into inferred inside results. Their warnings remain visible.
   The stage is retained through subsequent packet movement or detector dropouts.
8. Strip removal needs five fresh absent-strip observations spanning at least
   one second, with the strip area in view and not covered by a hand. Only hands
   near the packet count; a detection on a distant fixture does not block this
   step. If the full-bag detector fails, a visual feature reference of the same
   packet body supplies the inspection region. An empty bed or an off-camera
   strip area cannot prove removal. Rejected strip candidates remain uncertain.
9. Once strip removal is confirmed, mark **Sealed — inferred from strip removal**
   and proceed to OCR even if jewel placement remains unconfirmed. No separate
   closure model or further sealing wait is required. The PDF and JSON record
   sealing separately from each jewel's outcome, flag all unconfirmed placements
   for review, and retain their source recording. Sealing never verifies packet
   contents. The sealing start timestamp is the first accepted strip detection.
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

The first validated sealing-strip detection starts loading the number detector
and PaddleOCR in the background. The UI shows OCR loading/readiness. Preparation
does not read or confirm a number; three matching confident reads are still
required after sealing completes.

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
bag, strip, and hand models letterbox the original camera frame directly to
320×320. Masks decode in camera coordinates, and false-positive filters crop
from the original frame without an intermediate 960-pixel resize.

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

## Packet mouth experiment

Run `./run_packet_sealing.sh --mouth-lab` to open the packet-state collection
and classification lab. The existing packet ONNX detector supplies the full
packet bounding box automatically; no mouth ROI selection is needed. Leave
classification unticked to save manually labelled packet crops. Train on those
crops, then tick classification to display OPEN_CAMERA, OPEN_SIDE, CLOSED_FOLDED
or UNCLEAR predictions on the packet box. This experiment does not change the
sealing acceptance rule. See [Packet Mouth Lab](packet_mouth_lab/README.md).

### Pickup trigger experiment: selectable home-position logic

Home mode now also follows a provisional grip–rubbing–return sequence in both
`pickup_trigger_test.py` and `pickup_trigger_motion_class.py`. Select
**Home-position logic** or pass `--trigger-mode home`. The trained Motion
classification mode remains separate.

`GRIP_CANDIDATE` starts buffered recording. If gold is occluded, `PICKUP_PENDING`
preserves the candidate ID and follows the associated hand; disappearance alone
never verifies departure. Stable same-ID gold displacement or independent
empty-home evidence can establish `PICKUP_VERIFIED`. Other jewels reliably at
home are logged as supporting evidence, never used alone to assign identity.

For now, at least three hand direction reversals with at least 1.2 seconds of
meaningful movement in/around the detected stone produce `RUBBING_PROVISIONAL`.
The rule uses stone-relative thumb/index position, ignores small jitter, and
resets on missing stone/hand, unavailable scene alignment or gaps over 0.8 seconds. This
is a temporary hand-action rule, not trained OK/NOT_OK or verified physical contact.

Recording continues through home-verification delays and legacy cancellation.
A fresh same-ID jewel, released from the hand, stable on the saved bench ROI
outside the stone for at least three observations over 0.6 seconds resolves:

- `COMPLETED_PROVISIONAL`: departure observed, rule-based rubbing, stable return.
- `COMPLETED_INFERRED`: departure occluded, rule-based rubbing, same-ID return.
- `RETURNED_RUBBING_UNVERIFIED`: rubbing rule was not detected before return; this is an
  unverified action result, not proof that no rubbing happened. Check `rubbing_blocker`.
- `CANCELLED`: passing hand; jewel stayed at its candidate position.
- `REVIEW_REQUIRED`: missing/ambiguous evidence, unexplained occlusion and return,
  or an unresolved event lasting 120 seconds. Saved footage is retained.

The jewel can return to a new bench spot; each grip event has its own departure
baseline while the initial fixed home remains unchanged. A return after rubbing
does not cancel the event. JSONL observations include `workflow`, source labels,
captured rubbing measurements, event times and identity limitations. Existing
60-second clip segmentation continues without ending the sequence. Return identity
relies on the upstream SLIC tracker and continuous association; ID switches remain
a limitation. Initial hand-clear reference capture is still required.

Home mode accepts 25–150% of the reference gold-mask area for provisional
segmentation tracking. Masks under 65% are explicitly marked partial. A partial
COG cannot independently verify pickup or finish a return: departure additionally
needs stable mask area, same-ID continuity and coherent hand/jewel movement across
repeated observations, including sufficient hand travel. Return still requires
more complete gold support. The separate ArUco third mode also uses the 25% provisional mask gate.
`gold_measurement` reports measured area ratio and exact rejection/acceptance reason.

When the landmark tracker changes hand IDs, two consecutive observations of one
hand near the fresh same-ID gold mask can reconnect the existing event, even
after the old short reacquisition window. Two nearby hands remain ambiguous;
another active jewel's hand is reserved. `hand_status` and `hand_rebindings` make
this visible in workflow logs. Missing-hand, missing-stone, stone-change and
insufficient-motion blockers are separate; no generic hidden/no-rubbing claim is
substituted for them. The 25% setting needs live validation with partial masks.

Home mode now separately measures every nonempty gold segmentation before
permanent identity is required. Yellow `Gold-T#` markers show mask centroids,
including unknown pieces during initial assignment or lost IDs. Ring holes are
preserved; the mathematical centroid can lie inside a ring's empty centre.
Temporary tracks use unique spatial/area continuity, reset on gaps over 0.8 s,
and do not inherit individual identity through merges or ambiguous associations.
Two fresh unique upstream tracker-ID matches are required to display a J label;
otherwise the label stays `ID pending`. These are existing SLIC tracker matches,
not a newly trained identity model.

`GOLD_MOVING_WITH_HAND` records three coherent hand/mask motion observations over
0.45 s with trusted camera compensation. This is provisional gold movement, never
J1 pickup proof or lifting verification. Unknown masks get their own table rows;
per-frame `gold_masks` logs retain COG, temporary track, association and motion
state. This observer does not override named-jewel verification or automatically
assign an unknown mask to a home. Previous mode is unchanged; ArUco mode also includes this gold-mask observer.

Home-mode stability now reports three states: `STABLE`, `CAMERA_MOVED`, or
`INSUFFICIENT_EVIDENCE`. Reference features are distributed across the fixed
background; current hand, packet and gold regions are excluded. Independent
background descriptor matching can recover alignment when optical flow loses
its points. Too few points and poor matches are logged separately from measured
camera motion. A local feature cluster alone cannot establish camera alignment.

A trusted transform maps the current jewel position back into the original
scene coordinates. Thus camera motion alone cannot prove departure, while real
jewel motion can be verified with its home covered. Rubbing and stable-return
positions also use the compensated scene. The fixed-pixel home observer remains
paused during measured camera movement; its `reference_scene_uncertain` flag
does not veto the separate sequence when compensation is trustworthy.
Unknown or excessively changed alignment keeps recording/hand tracking active
but pauses verification; it never defaults to stable. Timing gaps remain a
separate failure. Logs and status show `scene_stability` with state, reason,
reference source, transform and trust, plus the workflow's stability summary.

Run `pickup_trigger_test.py` with the Jewel Tracker `.venv-slic` Python. The
**Previous logic** button selects the existing trigger (the default).
**Home-position logic** uses `model onnx/hand_landmarker.task` on CPU and a separate
pickup state machine. Install `requirements-pickup.txt` into `.venv-slic` first.
You can also start with `--trigger-mode home`. The old hand segmentation model
is no longer loaded by either Home-position or Fingertip mode.

Home mode captures each jewel's fixed segmentation mask, mask COG, position and
bench reference after 1.5 seconds of stable, freshly detected, uncovered jewels.
Thumb/index tip regions and distal finger segments near the current validated
gold mask start provisional recording immediately, with
one second of available pre-roll. Visible gold COG, displacement, velocity and
the associated hand are recorded in `observations.jsonl`. A fully hidden jewel
has no measured COG; its hand position is logged separately. Merged/crossing
hand associations and overlapping jewel detections are treated conservatively.
Fresh, sufficiently complete gold masks can supply COG measurements despite
IN_HAND / UNDER_HAND labels; missing or small partial masks cannot invent motion.
Proximity uses actual gold pixels, preserving bangle holes, rather than box
interiors or COG distance. A fixed-home mask is a provisional fallback when live
gold is unavailable. A pose hand near multiple jewels does not acquire an ID.
Matching finger-region and gold movement across three observations over at least
0.45 seconds can mark `PICKUP_DETECTED`; final `PICKUP_CONFIRMED` still waits for
independent empty-home verification. Passing-hand motion alone cannot do this.

Landmarks provide proximity, not physical contact or a pixel-accurate hand mask.
A padded hand/wrist area pauses home verification when it overlaps original gold
support. The half-visible-home path is disabled for landmark detections because
keypoints cannot establish which home pixels are exposed. Full frames still go
to the model; outside-ROI hands are filtered afterward. Thumb/index approach
alone is a candidate, never pickup confirmation.

The original-home observer independently confirms empty-home evidence when the
home is uncovered, the camera is stable, no packet or gold covers the home, and
the home matches visible empty bench across at least three observations spanning
0.45 seconds. This is supporting evidence rather than a mandatory sequence gate:
the workflow can also verify sustained same-ID displacement. A jewel still at
home cancels a passing-hand candidate only when no departure or rubbing evidence
was recorded. Packet overlap is an optional destination hint;
it does not prove insertion. Rubbing-stone destination confirmation is not
implemented in this isolated test.

Changing modes starts a fresh assignment in a new `reset_###` folder, preserving
previous references, event logs and clips. Session/event logs record the mode.
The main packet packing and acid workflows are unaffected.

## Strip-removal troubleshooting

During removal, the phase label names the current blocker: packet reference lost,
strip area not visible, hand covering strip area, strip still detected, or strip
prediction uncertain. It shows accepted absence observations (`0/5`) and clear
observation time (`0.0/1.0 s`). A source-time gap over 2.5 seconds resets progress.
Slow inference, stale frames and camera interruptions also reset it.

When the original packet feature reference can be matched, its strip attachment
area moves with that reference. Strip predictions and hand clearance are checked
there, with a surrounding margin, rather than across the entire packet. A loose
strip outside this zone is logged as `outside_attachment_area`; this is a spatial
exclusion, not a separate model identifying loose versus attached material. A
strip still in the zone blocks confirmation. If no feature reference can be
prepared, the full packet remains the conservative inspection scope. If a prepared
reference is lost, full-bag detection alone no longer bypasses its visibility
check. Return the packet with its original strip area visible to recover.

A prediction marked uncertain may pause existing progress for up to 0.8 seconds
since the last clear observation. It adds neither observations nor clear time.
Prolonged uncertainty, hands in the area, lost context/visibility and accepted
strip detections reset progress. Five clear absence observations and at least
one second accumulated between consecutive clear observations remain required.

Each output directory retains `sealing_diagnostics.jsonl` with source timestamps,
blocker, progress, gap resets, inspection scope, candidate overlap/filter scores,
and hand-pixel counts. Camera/inference failures are logged separately. The
`sealing_debug/` directory saves the first source scene for each blocker. These
files remain after `.working` cleanup. `evidence.json` includes blocker counts and
the final removal check; the PDF also displays the final check. The first saved
scene is a diagnostic example, not necessarily the last occurrence of that blocker.


## Bag-filter acceptance cutoff

The primary bag detector still requires 0.50 confidence. The separate bag
false-positive classifier now accepts scores at or above 0.20 (previously 0.60).
Both packet-only detection and live inspection use the same cutoff. Strip
thresholds, bench ROI checks and the exactly-one-accepted-packet rule are unchanged.
The constructor accepts `bag_filter_threshold` for controlled comparisons.

The saved-session check in `outputs/bag_filter_threshold_check.json` compared 35
packet-scene samples: 3 were accepted at 0.60, 32 at 0.30, and 35 at 0.20. One
known non-packet scene was rejected at 0.20. Samples include compressed video
and JPEG scenes from the same session, not an independent accuracy evaluation.
Broader negative testing is still needed before claiming a false-positive rate.
A 0.20 classifier score is an acceptance cutoff, not 20% detector confidence.

Sealing diagnostics now include each raw bag candidate's detector confidence,
bench ROI overlap, filter score, acceptance/rejection decision and the active
filter threshold. This distinguishes a detector miss from a filter rejection.


## Current sealing flow: no saved-reference requirement

The live application now checks strip absence inside the currently detected
packet mask. It does not prepare, match or require the saved strip-area reference.
An earlier reference loss cannot block a current packet detection. Hands must
be clear, and five qualifying absence observations with at least one second
of clear observation time confirm inferred sealing after the strip was first seen.
Missing current packet detection pauses the stage and resets the absence counter;
its blocker is `packet_not_detected`. OCR follows confirmation as before.
This section supersedes the saved-reference behavior described earlier.

## Standalone third trigger: ArUco hand-to-stone

Home and ArUco modes now connect the mask-only observer to candidate recordings.
A unique temporary track previously matched to a named candidate may continue
supplying provisional position/hand association when the named detector drops out.
`MOVEMENT_OBSERVED_ID_PENDING` distinguishes observed mask movement from verified
named departure. Fresh named evidence is required twice before this path can
verify the permanent ID. Merges, conflicting links and ambiguous masks do not
bridge identities. `temporary_track`, `departure_observed` and measurement source
are recorded in the workflow.

Palm/wrist support augments fingertip proximity for continuity. Mask tracks and
temporal votes survive missed observations for at most 0.8 seconds; missing
centroids are logged as absent and are never drawn as real measurements. Gaps or
contradictory observations still reset evidence. Unassigned masks can start their
own buffered recording and produce `UNASSIGNED_RUBBING_PROVISIONAL` based on the
existing hand-to-stone rule. This identifies an unassigned hand-action event,
not a verified jewel identity or physical rubbing contact. Unknown-event logs
always keep `jewel_id=null`, `identity_verified=false`, and `pickup_verified=false`.
Their clips use `JGold-T#` filenames. Losing the temporary track for two seconds
closes its recording for review. `unassigned_events` and mask workflow labels
expose the action even while named events remain pending.

The third button in both pickup scripts is now **ArUco hand-to-stone**. Existing
Home-position mode stays separate; the motion-classification experiment retains
its fourth mode. The CLI key `cog` is retained for launch compatibility:

```bash
../jewel_tracker/jewel_tracker/.venv-slic/bin/python pickup_trigger_test.py --trigger-mode cog --source 0 --autostart
```

Place one flat, rigidly fixed ArUco marker near the top-left of the bed with its
white margin visible. Detection receives the full frame, including markers outside
the jewellery ROI. Common 4x4/5x5/6x6/7x7 dictionaries (up to 1000 IDs), original
ArUco and AprilTag 16h5/25h9/36h10/36h11 are tried automatically. One unambiguous marker must have at least 24-pixel
sides and three stable observations. That dictionary/ID is then locked; a different
marker is not substituted. Homes are saved only after marker acquisition and the
usual hand-clear jewel stability checks. Reset reacquires marker and homes.

The marker's four corners supply a planar homography to the initial bed view.
Jewel/hand movement and return use this transform; the unchanged pixel-based home
observer pauses during measured camera movement. Missing, clipped, too small or
ambiguous markers never silently reuse an old transform: fixed-background alignment
is the fallback. If neither reference is trustworthy, recording continues while
verification waits. Status text and JSONL `aruco` entries expose marker ID,
dictionary, corners and rejection reason. A marker outline is drawn on the preview.

The sequence is hand–jewel proximity → buffered recording → follow associated hand
while jewel is hidden → sustained reversals near stone → provisional rubbing →
identity/return resolution. Reliable gold departure can verify pickup with home
covered. Hand motion alone never establishes a named jewel's departure. Independent
gold-mask COGs and temporary `Gold-T#` tracks remain available when J IDs are missing.

The stone detector still runs on the full frame. If the stone is briefly hidden,
its last aligned location may be reused for at most two seconds while the marker
is currently visible and trusted. Moving the stone independently requires a new
stone detection. Rubbing remains a temporary hand-motion rule, not trained action
classification or verified physical contact. The third mode loads gold, hand
landmarks, stone and ArUco; it does not load the packet model.

A single corner marker gives weaker accuracy far away from it. Blur, lens distortion
and corner error can exaggerate full-bed transforms; lifted jewellery also violates
the bed-plane assumption. Live validation is still required. No physical dimensions
or calibrated 3D pose are inferred. `--cog-threshold` remains accepted for older
launchers; the new sequence uses the documented home-flow departure thresholds.


### Marker-only ROI selection and diagnostics

Start the camera/video, select **ArUco hand-to-stone**, then click **Set marker
ROI**. Drag around the complete marker including its white margin and click OK.
The selection is normalized to the camera image, saved in local `aruco_roi.json`,
and shared by both pickup scripts. Changing it resets ArUco-mode references while
preserving recordings; keep hands clear. Other modes are unaffected. **Auto marker
ROI** restores the top-left search area. Crops receive 10% padding (minimum 8
capture pixels) and are enlarged up to 3x when small. Corners are mapped back to
full capture coordinates before alignment. Gold, hand and stone model input remains
full-frame. The marker search area is outlined separately on the preview.

Diagnostics distinguish no supported code decoded, wrong locked ID, small/skewed/
clipped markers and multiple candidates. `decoded`, `rejections`,
`rejected_quadrilaterals`, `search_roi` and `upscale` are logged. Rejected
quadrilaterals alone do not prove a dictionary mismatch. The physical marker in
the saved November 18 screenshots decoded as **AprilTag 36h11, ID 1**; the earlier
ArUco-only search therefore missed it. This was verified on saved screenshots,
not a new live-camera measurement. Initial acquisition still needs three stable
observations and marker sides of at least 24 capture pixels.


### Home and ArUco processing without a scene-stability gate

Home mode now processes raw-frame movement without checking background stability.
Its status is `NOT_CHECKED` / `stability_gate_disabled`; it does not claim that the
camera was verified stationary. ArUco mode still compensates coordinates when the
marker transform is usable, but a missing/rejected transform falls back to raw
coordinates without pausing the sequence. The marker needs three matching ID
observations, not stationary corners. Initial clear-home capture, valid marker
geometry, jewel identity, partial-mask checks, motion consistency, timing-gap and
stable-return checks remain. Camera movement can now contribute apparent movement
when compensation is unavailable. Existing Home and third-mode processing no
longer use the earlier background-alignment fallback described above.


### Separate pickup, rubbing and return results

The experiment table now has separate Pickup, Rubbing and Return columns plus an
`E#` phase/event column. A verified pickup stays `VERIFIED` for that event even
when rubbing is unverified or the jewel has returned. Observed/verified pickup
times come from the current event rather than stale core-observer timestamps.
New interactions increment the event number, preserve the previous summary, and
archive the prior event snapshot in `HomeRubbingFlow.history`.

A candidate's departure baseline is its last reliably confirmed bed position,
falling back to the initial fixed home. A partial mask measured after gripping
cannot redefine that baseline. Reliable same-ID relocation can verify pickup
without rubbing. Cancellation now requires a full named observation at the
confirmed starting position with no departure/rubbing evidence. A stationary
jewel elsewhere with insufficient departure evidence goes to review; it is not
labelled as having stayed at home. Per-frame logs include `stone_box`, and each
workflow includes `stone_observation`, independent status fields, event number,
origin source and pickup evidence timestamps. Unknown gold events always display
`UNASSIGNED` pickup rather than inheriting an internal provisional result.
