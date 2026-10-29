# PothPradarshak

**Object Detection and Navigation Assistance for Visually Impaired Individuals**
University thesis prototype — real-time YOLO detection with Bengali and English voice guidance.

> **Research prototype.** This system provides experimental computer-vision
> assistance. It must not be relied upon as the sole navigation aid for
> real-world mobility.

---

## Current status — read this first

| Component | State |
|---|---|
| Real-time application (camera → detection → position → voice → UI) | **Working** |
| Input modes: Live Camera / Upload Image / Upload Video | **Working** |
| Bengali + English voice guidance, cooldown, priority | **Working** |
| Frontend matching the supplied UI/UX design | **Working** |
| GPU acceleration (RTX 3050 Ti, CUDA) | **Working** |
| Dataset selection + de-duplication | **Working** (2,787 images) |
| Auto-generated candidate labels | **Working** (unverified) |
| Human-verified labels | **Done** — 2,782 images exported |
| Trained thesis model (`best.pt`) | **Trained** — 100 epochs, see results below |
| Indoor object detection (2nd model) | **Working** |
| Analyse uploaded image / video | **Working** |

The application now loads the trained thesis model automatically. The
placeholder-COCO fallback still exists for a fresh clone with no `best.pt`,
and is announced in the log, on the video overlay and in a UI banner so it can
never be mistaken for a thesis result.

## Results — trained model, held-out test split

100 epochs, YOLO11n, 640 px, batch 8, RTX 3050 Ti.
**Overall: precision 0.645 · recall 0.404 · mAP@50 0.419 · mAP@50-95 0.297**

| class | P | R | mAP@50 | mAP@50-95 | test boxes |
|---|---:|---:|---:|---:|---:|
| person | 0.811 | 0.749 | **0.837** | 0.600 | 878 |
| parked_vehicle | 0.758 | 0.688 | **0.781** | 0.528 | 1360 |
| curb | 0.413 | 0.450 | 0.359 | 0.292 | 20 |
| street_obstacle | 0.404 | 0.267 | 0.295 | 0.197 | 99 |
| pole | 0.263 | 0.222 | 0.268 | 0.221 | 9 |
| tree | 0.372 | 0.262 | 0.237 | 0.138 | 65 |
| stairs | 0.785 | 1.000 | 0.995 | 0.697 | **2** |
| pothole | 1.000 | **0.000** | **0.000** | 0.000 | **1** |
| open_drain | 1.000 | **0.000** | **0.000** | 0.000 | **1** |

**Read the last three rows honestly.** `stairs` at 0.995 is not a good result —
it is two test boxes, which is noise, not evidence. `pothole` and `open_drain`
score **zero recall**: the model finds none of them, because it was trained on
17 and 12 instances respectively. Precision 1.000 there is an artefact of never
making a positive prediction at all.

The model is genuinely good at `person` and `parked_vehicle` (the two classes
with thousands of examples) and genuinely blind to the fall hazards. Fixing
that needs annotation, not more epochs — see the class-imbalance section below.

---

## Install

```bash
pip install -r requirements.txt
```

**For GPU training** (strongly recommended — 11× faster here), install the CUDA
build of PyTorch rather than the default CPU wheel:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

---

## Run the application

```bash
python app/main.py
```

Then open <http://127.0.0.1:8000> and press **Start Camera**.

Useful flags:

```bash
python app/main.py --camera 1                 # a different camera index
python app/main.py --camera clip.mp4          # a video file (loops, real-time)
python app/main.py --camera http://192.168.0.101:8080/video   # phone as camera
python app/main.py --language en              # start in English (default: Bengali)
python app/main.py --device cpu               # force CPU
python app/main.py --host 0.0.0.0             # reach it from your phone
python app/main.py --autostart                # skip pressing Start
```

A video-file source loops forever and plays at its own frame rate, which makes
it a reproducible stand-in for the camera when demonstrating the system.

### Getting a live camera

**This laptop has no built-in webcam.** The only camera devices are **Camo**
(phone-as-webcam) and an NVIDIA Broadcast virtual camera. With no phone
attached, those virtual cameras still open and still stream — a grey
placeholder graphic or an NVIDIA "camera disabled" splash. The app then runs
at full speed and detects nothing, which looks exactly like a broken model but
is not.

Check what your machine can actually see:

```bash
python -m backend.camera
```

It lists every index, which backend works, and whether the image is real or
blank, then tells you the exact command to use.

Three ways to get a live feed:

| Method | Command | Notes |
|---|---|---|
| **Camo** | start Camo on the phone **and** Camo Studio on the PC, then `python app/main.py --camera 0` | phone becomes the camera |
| **IP webcam** | install an IP-webcam app, then `python app/main.py --camera http://<phone-ip>:8080/video` | phone and laptop on the same Wi-Fi; note the `/video` suffix |
| **Video file** | `python app/main.py --camera clip.mp4` | reproducible, no hardware needed |

A phone camera is genuinely the better choice for this thesis anyway: held at
chest or head height it matches a blind pedestrian's viewpoint, which a laptop
webcam never does. Run with `--host 0.0.0.0` and you can watch the dashboard on
the phone itself while it films.

---

## The annotation pipeline (the remaining work)

The `image/` folder contains 21,359 images and **no annotations whatsoever**.
Detection training is impossible without boxes, so labels must be produced.
The pipeline is four steps:

```
 image/ ──▶ build_annotation_set.py ──▶ autolabel.py ──▶ review_labels.py ──▶ prepare_dataset.py ──▶ train.py
  21,359         2,787 deduplicated      candidate boxes    YOU verify           train/val/test
  raw images     Dhaka footpath imgs     (machine, unverified)                   splits
```

### Step 0 — select and de-duplicate  *(done)*

```bash
python scripts/build_annotation_set.py
```

Selects the Dhaka footpath group (numeric `.jpg`, 256×256), removes
near-duplicates by perceptual hash, and writes 2,787 images to
`annotation_input/`. De-duplicating **before** annotating avoids labelling the
same scene twice and makes train/test leakage structurally impossible.

### Step 1 — generate candidate labels  *(done)*

```bash
python scripts/autolabel.py                # whole set
python scripts/autolabel.py --limit 50     # trial run
python scripts/autolabel.py --report       # summarise
```

Runs an open-vocabulary detector (YOLO-World v2) with descriptive prompts per
class, maps its output back to the 9 thesis classes, and filters the two
mechanical failure modes (whole-image boxes, cross-prompt duplicates).

**These labels are machine guesses, not ground truth.**

### Step 2 — verify every image  ← **you are here**

```bash
python scripts/review_labels.py            # opens http://127.0.0.1:8009
```

| Key | Action |
|---|---|
| `A` / `Enter` | accept image as correct, next |
| `D` | delete selected box |
| `1`–`9` | set class of selected box |
| `N` | skip (decide later) |
| `Backspace` | previous image |
| drag | draw a new box |

Images are served **hardest-first**: candidates for `pothole`, `open_drain`
and `curb` come before easy ones, so if review time runs short, the classes
that matter most for navigation are the ones that got done.

Progress: `python scripts/review_labels.py --stats`
Export when done: press **Export**, or `--export-only` → `annotated/`

### Step 3 — build splits and train

```bash
python scripts/prepare_dataset.py     # 70/20/10 train/val/test
python scripts/train.py               # writes runs/train/exp/weights/best.pt
python scripts/validate.py            # precision, recall, mAP50, mAP50-95, confusion matrix
```

`train.py` **refuses to run** on a dataset with zero boxes rather than silently
producing a model that detects nothing. Once `best.pt` exists the application
finds it automatically and the placeholder warning disappears.

Useful training flags:

```bash
python scripts/train.py --model yolo11s.pt --epochs 150 --batch 8
python scripts/train.py --imgsz 960 --name exp_highres
python scripts/train.py --resume runs/train/exp/weights/last.pt
```

On a 4 GB GPU start with `yolo11n.pt` at `--batch 8` or `--batch -1` (auto).

---

## How well does auto-labelling actually work?

Measured on this dataset, not assumed. This matters for the methodology
chapter — and it is the reason step 2 cannot be skipped.

Measured over all 2,787 images: **24,010 candidate boxes** in 5.6 min on the
RTX 3050 Ti, with 3,454 mechanical errors filtered out automatically
(2,383 cross-prompt duplicates, 1,009 whole-image boxes, 62 specks).

| Class | Candidates found | Share | What you must do |
|---|---:|---:|---|
| `parked_vehicle` | 13,328 | 55.5 % | spot-check, delete false positives |
| `person` | 8,820 | 36.7 % | spot-check |
| `street_obstacle` | 833 | 3.5 % | add the misses |
| `tree` | 606 | 2.5 % | add the misses |
| `curb` | 204 | 0.8 % | **verify carefully — mostly wrong** |
| `pole` | 153 | 0.6 % | add the misses |
| `stairs` | 23 | 0.1 % | **draw by hand** |
| `open_drain` | 22 | 0.1 % | **draw by hand** |
| `pothole` | 21 | 0.1 % | **draw by hand** |

**Read that distribution carefully — it is the central methodological problem
of this thesis.** Two classes account for **92 %** of all candidates, while the
three classes the navigation engine treats as most dangerous account for
**0.3 %** between them.

Training on this as-is would produce a competent person-and-vehicle detector
that has *never seen a pothole* — a navigation aid for blind pedestrians that
is blind to exactly the hazards that injure them. That model would score a
respectable overall mAP while being useless for its stated purpose, which is
precisely the kind of result an examiner is trained to find.

An open-vocabulary model has no clean visual prototype for a hole in pavement,
and cannot separate one from a shadow, a patch or an ordinary paving seam.
**The hazard classes are yours to draw.** Aim for at least 100–200 verified
instances of each of `pothole`, `open_drain`, `stairs` and `curb` before
training, and expect that to dominate your review time. The review tool serves
these images first for exactly this reason.

The three classes the detector cannot find are precisely the three the
navigation engine treats as most dangerous. An open-vocabulary model has no
clean visual prototype for a hole in pavement, and cannot separate one from a
shadow, a patch or an ordinary paving seam. **The hazard classes are yours to
draw.** Budget review time accordingly.

---

## Indoor objects (chair, table, bed …)

The thesis model is trained on Dhaka footpaths, so indoors it correctly reports
an empty scene. To make the system useful indoors, `backend/detector.py` runs a
**second, COCO-pretrained model alongside it** and takes only these classes:

`chair` `couch` `bed` `dining table` `tv` `toilet` `sink` `refrigerator`
`oven` `microwave` `potted plant` `bench`

Toggle it in the UI under **সহায়তা সেটিংস → ঘরের ভিতরের বস্তু** (Indoor Objects).

Two rules keep the fusion honest:

- The indoor model is **excluded from every class the thesis model covers** —
  notably `person` — so the two never double-report one object.
- On any overlap above 0.55 IoU the **thesis model wins**, because it is the
  one trained on this thesis's data.

Indoor objects are announced like any other obstacle, and score *below* fall
hazards in the priority engine: a chair is worth mentioning, but never more
urgent than a pothole.

**State this in the thesis:** these indoor classes are COCO-pretrained and
carry **no thesis-measured mAP**. They are a deployment convenience, not a
research contribution. `door` is unavailable — COCO has no door class, so
honest support for it needs its own annotated set.

---

## Input modes

The dashboard has three input modes, selected by the segmented control under
the page title. **They share one detection stack** — the same trained model,
the same thresholds, the same spatial analyser, navigation engine and voice
engine. Mode changes only where the *frames* come from:

```
                       ┌──────────────┐
   Live Camera ───┐    │              │
   Upload Video ──┼───▶│  CameraSource│──▶ pipeline ──▶ detector ──▶ spatial
   Upload Image ──┼─┐  │              │        ▲            │
                  │ │  └──────────────┘        │            ▼
                  │ │                          │       navigation
                  │ └──▶ MediaAnalyzer ────────┘            │
                  │      (single frame)                     ▼
                  │                                    voice + UI
```

| Mode | Source | Controls |
|---|---|---|
| **লাইভ ক্যামেরা** | webcam / phone / IP stream | Start, Stop, Detection ON/OFF, Emergency Stop |
| **ছবি আপলোড** | one still image | Choose Image, Detect, Clear |
| **ভিডিও আপলোড** | recorded clip | Choose Video, Load, Play, Pause, Restart, Stop, Loop |

### Why video mode reuses the camera path

An uploaded clip becomes the pipeline's **source**, exactly like a webcam.
`CameraSource` already paced video files at their native FPS and looped them,
so video mode needed playback control — not a second detection implementation.

That is the whole reason frames stream **as they are processed** rather than
the clip being analysed in full and replayed afterwards: there is only one
loop, and it is the live one. It also means MJPEG streaming, the detection
table, FPS, inference time, navigation and voice all work in video mode with
no video-specific code, and a bug fixed in one mode is fixed in all three.

Playback control lives in `CameraSource.pause/resume/restart/playback()` and is
deliberately a **no-op on a live camera** — reality cannot be paused, and
pretending otherwise would mislead the caller. `/api/playback` returns a clear
error instead.

---

## Analyse an image or video

The right-hand panel **ছবি / ভিডিও বিশ্লেষণ** accepts a file and runs the same
chain over it — no camera required. Useful when the camera is unavailable, and
essential for a defence, where you need the same result every time.

```
POST /api/analyze   file=<image|video>  language=bn|en  speak=true
```

- **Images** → annotated JPEG, per-object position/proximity, one spoken instruction.
- **Videos** → sampled at ~2 frames/second (capped at 90 frames), producing a
  **timeline of instructions with timestamps**. Consecutive duplicates are
  collapsed, for the same reason the live system has a cooldown: a blind user
  needs the sequence of events, not 1,800 sentences.

Measured on a 90-second clip: 90 frames analysed in 5.4 s, 329 detections,
39 distinct instructions.

It reuses the live pipeline's detector, spatial analyser and navigation engine,
so an uploaded file is judged by exactly the same model and thresholds as the
camera — and it does not touch the camera, so a live session keeps running.

---

## Architecture

```
Camera ─▶ Frame ─▶ YOLO ─▶ Spatial ─▶ Navigation ─▶ Voice ─▶ UI
                          analysis     engine        engine
```

```
backend/
├── camera.py        threaded capture; newest-frame-wins, auto-reconnect, video-file loop
├── detector.py      the only module that touches Ultralytics; inference + overlay
├── spatial.py       bbox → LEFT/CENTER/RIGHT + NEAR/MEDIUM/FAR
├── navigation/      priority, event detection, cooldown — decides what to say and when to stay quiet
├── voice/           message_generator.py (EN/BN sentences) + tts.py (non-blocking speech)
├── pipeline.py      the worker thread joining all of the above
├── config.py        thread-safe runtime settings + model auto-discovery
└── main.py          FastAPI: MJPEG stream + JSON state API
frontend/            the UI, matching the supplied design
scripts/             dataset, annotation, training and evaluation tooling
app/main.py          entry point
```

Each stage is independently testable; the detector knows nothing about
navigation, and the navigation engine knows nothing about YOLO.

### Threading

Three threads, deliberately: camera capture, the vision pipeline, and TTS
synthesis. Voice never blocks inference, inference never blocks HTTP, and the
camera always holds the newest frame — for a navigation aid a fresh frame
matters far more than processing every frame.

---

## How the guidance works

**Position** — the frame is split into three vertical zones (default 0–33 %,
33–66 %, 66–100 %, configurable). An object's zone comes from its box centre.

**Proximity** — from box area relative to frame area: NEAR / MEDIUM / FAR.
This is **approximate proximity, not distance**. A single RGB camera cannot
measure metres, so the UI never claims to. `spatial.py` has a `DepthProvider`
hook: supply MiDaS or a stereo/ToF source and proximity becomes real depth
with no other code changing.

**Priority** — a score, not a magic number: danger class +100, CENTER +50,
NEAR +30 / MEDIUM +12, plus confidence as a tie-break. A pothole ahead
therefore always outranks a chair to the left.

**When to speak** — speech is triggered by *changes*, not by presence:
a new object, proximity worsening, or something moving into your path. A pole
that has been on your left for 20 seconds is not news. On top of that sits a
per-object cooldown and a global minimum gap, both configurable. An escalation
to NEAR overrides the cooldown, because that is the moment you most need to
hear it.

Critically, the engine compares against the state at the **last announcement**,
not the last frame — otherwise an escalation suppressed by the cooldown would
be consumed and silently lost.

**Bengali** — `voice/message_generator.py`. Bengali is Subject-Object-Verb, so
sentences are assembled tail-last with আছে at the end, and it uses numeral
classifiers: একজন for people, একটি for things. "একটি মানুষ" is wrong and jarring
to a native speaker, so animacy is tracked per class.

```
বামে চেয়ার আছে              সামনে খুব কাছে একজন মানুষ আছে
সাবধান! সামনে খুব কাছে একটি গর্ত আছে
```

---

## Voice / TTS on this machine

Windows SAPI has **no Bengali voice installed here**, so `pyttsx3` physically
cannot speak Bengali. The engine therefore picks:

- **English** → `pyttsx3`, offline, instant
- **Bengali** → `edge-tts` (`bn-BD-NabanitaNeural`), free, no API key,
  **requires internet**
- **Fallback** → browser Web Speech API, then silent (text still shown)

To get offline Bengali: Windows Settings → Time & language → Language & region
→ Add a language → বাংলা (বাংলাদেশ) → tick *Speech*. It is then picked up
automatically.

TTS failure never crashes the app and never blocks the camera.

---

## Testing

```bash
python scripts/inference.py --source annotation_input/1000.jpg   # single image
python scripts/inference.py --source annotation_input/ --save    # a folder
python scripts/validate.py                                       # metrics on the test split
python scripts/analyze_dataset.py                                # dataset composition
python scripts/review_labels.py --stats                          # annotation progress
```

---

## Configuration

Everything tunable lives in `backend/config.py` and most of it is editable live
from the UI or the API:

| Setting | Default | Meaning |
|---|---|---|
| `confidence` | 0.50 | detection threshold |
| `announce_cooldown` | 3.0 s | per-object repeat gap |
| `global_min_gap` | 1.2 s | minimum gap between any two utterances |
| `zone_left_edge` / `zone_right_edge` | 0.33 / 0.66 | LEFT/CENTER/RIGHT boundaries |
| `near_area_ratio` | 0.15 | box area fraction counted as NEAR |
| `language` | `bn` | Bengali by default |
| `max_spoken_objects` | 1 | objects mentioned in one utterance |

---

## Known limitations

State these plainly in the thesis rather than letting an examiner find them:

1. **No trained thesis model yet.** Annotation is incomplete; the app runs on
   COCO weights until it is done.
2. **Proximity is not distance.** Box area only. Comparable between objects of
   similar real size, never across classes.
3. **The hazard classes cannot be auto-labelled** and must be drawn by hand.
4. **Bengali TTS needs internet** on this machine.
5. **No object tracking.** Guidance keys on (class, position), so two people on
   your left are one piece of news, and a specific object is not followed
   across frames.
6. **Single camera, single viewpoint.** No stereo, no IMU, no GPS.
7. **Daylight data.** The selected images are daytime; night performance is
   untested.
8. **Latency.** ~25 FPS on a laptop RTX 3050 Ti; a real aid would need an
   embedded target and a latency budget measured end-to-end to the ear.
