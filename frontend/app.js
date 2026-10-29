/* ==========================================================================
   Navigation Assist — frontend controller
   --------------------------------------------------------------------------
   Every value rendered here comes from the backend. There is no simulated
   detection, no fabricated frame rate and no invented distance anywhere in
   this file. When the backend cannot supply a value the UI shows an em dash.

   Organised as small modules against the existing API, which is unchanged:
       api        thin fetch wrapper
       i18n       English / Bengali interface strings
       audio      queued speech playback with autoplay unlock
       Mode       input-mode switching
       Viewport   camera / image / video display
       Safety     safety banner
       Guidance   instruction card and direction row
       Objects    detected-object list
       Playback   video transport
       Settings   settings sheet
   ========================================================================== */

"use strict";

const POLL_MS = 250;          // 4 Hz — responsive without loading the CPU

/* ------------------------------------------------------------------ i18n */

const I18N = {
  en: {
    tagline: "Ready to help you navigate",
    modeLive: "Camera", modeImage: "Image", modeVideo: "Video",
    cameraOff: "Camera is off", cameraOffHint: "Press Start to begin detection",
    imageIdle: "No image selected", imageIdleHint: "Choose an image to analyse",
    videoIdle: "No video loaded", videoIdleHint: "Choose a recorded video to analyse",
    starting: "Starting camera…", analysing: "Analysing image…", loadingVideo: "Loading video…",
    live: "LIVE", objects: "Objects", speed: "Speed", model: "Model",
    start: "Start Camera", stop: "Stop", stopAll: "Stop All",
    play: "Play", pause: "Pause", restart: "Restart",
    playing: "Playing", paused: "Paused", finished: "Finished",
    selectImage: "Select an image", selectVideo: "Select a recorded video",
    dropHint: "or drop a file here", analyse: "Analyse", clear: "Clear",
    loadVideo: "Load Video", loop: "Repeat",
    guidanceLabel: "Current guidance", pathClear: "Path is clear",
    detectionOff: "Detection is off", notRunning: "Not started",
    voiceLabel: "Voice guidance", detectedLabel: "Detected",
    noObjects: "Nothing detected yet",
    on: "On", off: "Off",
    left: "Left", ahead: "Ahead", right: "Right",
    near: "Appears close", medium: "Nearby", far: "Further away",
    safeClear: "Path appears clear", safeAttention: "Object nearby",
    safeWarning: "Obstacle ahead", safeCritical: "Stop — obstacle directly ahead",
    settings: "Settings", detection: "Object detection", indoor: "Indoor objects",
    confidence: "Confidence threshold",
    confidenceHint: "Lower detects more objects but with more mistakes.",
    cooldown: "Repeat delay",
    cooldownHint: "How long before the same object is announced again.",
    announceFar: "Announce distant objects",
    proximityNote: "Distances are estimated from how large an object appears. They are approximate and are not measured.",
    disclaimer: "Research prototype. Do not rely on this as your only navigation aid.",
    retry: "Retry",
    errCamera: "Camera unavailable",
    errCameraBody: "Check that the camera is connected and not in use by another app, then try again.",
    errBlank: "Camera is sending a blank image",
    errBlankBody: "If this is a virtual camera, connect a phone or source to it, or choose a different camera.",
    errImage: "Could not analyse the image",
    errVideo: "Could not process the video",
    errGeneric: "Something went wrong",
    audioBlocked: "Tap anywhere once to enable voice",
    srcDevice: "This device", srcServer: "Server camera",
    switchCam: "Switch Camera",
    camAsking: "Asking for camera permission…",
    camDenied: "Camera permission denied",
    camDeniedBody: "Allow camera access for this site in your browser settings, then press Start again.",
    camNone: "No camera found",
    camNoneBody: "This device does not report a usable camera.",
    camInsecure: "Camera needs a secure connection",
    camInsecureBody: "Open this page over https:// (your Cloudflare Tunnel address) — browsers block the camera on plain http.",
    camUnsupported: "This browser cannot open the camera",
    camUnsupportedBody: "Try Chrome on Android, or Safari on iPhone.",
    camBusy: "Camera is in use",
    camBusyBody: "Another app or tab is using the camera. Close it and try again.",
    detecting: "Detecting objects…",
    serverDown: "Detection server is unavailable",
    serverDownBody: "The page loaded but the detection server did not answer. Check the tunnel is still running.",
    placeholderModel: "Running a general-purpose model, not the trained thesis model."
  },
  bn: {
    tagline: "আপনাকে পথ চলতে সাহায্য করতে প্রস্তুত",
    modeLive: "ক্যামেরা", modeImage: "ছবি", modeVideo: "ভিডিও",
    cameraOff: "ক্যামেরা বন্ধ", cameraOffHint: "শুরু করতে স্টার্ট চাপুন",
    imageIdle: "কোনো ছবি নির্বাচন করা হয়নি", imageIdleHint: "বিশ্লেষণের জন্য ছবি বেছে নিন",
    videoIdle: "কোনো ভিডিও লোড করা হয়নি", videoIdleHint: "বিশ্লেষণের জন্য ভিডিও বেছে নিন",
    starting: "ক্যামেরা চালু হচ্ছে…", analysing: "ছবি বিশ্লেষণ হচ্ছে…", loadingVideo: "ভিডিও লোড হচ্ছে…",
    live: "লাইভ", objects: "বস্তু", speed: "গতি", model: "মডেল",
    start: "ক্যামেরা চালু", stop: "বন্ধ", stopAll: "সব বন্ধ",
    play: "চালু", pause: "বিরতি", restart: "পুনরায়",
    playing: "চলছে", paused: "বিরতি", finished: "শেষ",
    selectImage: "ছবি নির্বাচন করুন", selectVideo: "ভিডিও নির্বাচন করুন",
    dropHint: "অথবা এখানে ফাইল ছাড়ুন", analyse: "বিশ্লেষণ", clear: "মুছুন",
    loadVideo: "ভিডিও লোড", loop: "পুনরাবৃত্তি",
    guidanceLabel: "বর্তমান নির্দেশনা", pathClear: "সামনের পথ পরিষ্কার",
    detectionOff: "শনাক্তকরণ বন্ধ", notRunning: "শুরু করা হয়নি",
    voiceLabel: "কণ্ঠস্বর নির্দেশনা", detectedLabel: "শনাক্ত হয়েছে",
    noObjects: "এখনো কিছু শনাক্ত হয়নি",
    on: "চালু", off: "বন্ধ",
    left: "বামে", ahead: "সামনে", right: "ডানে",
    near: "খুব কাছে", medium: "কাছে", far: "দূরে",
    safeClear: "সামনের পথ পরিষ্কার", safeAttention: "কাছে বস্তু আছে",
    safeWarning: "সামনে বাধা আছে", safeCritical: "থামুন — সামনেই বাধা",
    settings: "সেটিংস", detection: "বস্তু শনাক্তকরণ", indoor: "ঘরের ভিতরের বস্তু",
    confidence: "আস্থার সীমা",
    confidenceHint: "কম রাখলে বেশি বস্তু শনাক্ত হয়, তবে ভুলও বাড়ে।",
    cooldown: "পুনরাবৃত্তির বিরতি",
    cooldownHint: "একই বস্তু আবার বলার আগে কত সময় অপেক্ষা করবে।",
    announceFar: "দূরের বস্তুও বলুন",
    proximityNote: "দূরত্ব বস্তুর আপাত আকার থেকে অনুমান করা হয়। এটি আনুমানিক, মাপা নয়।",
    disclaimer: "গবেষণা প্রোটোটাইপ। এটিকে একমাত্র চলাচল সহায়ক হিসেবে ব্যবহার করবেন না।",
    retry: "আবার চেষ্টা",
    errCamera: "ক্যামেরা পাওয়া যায়নি",
    errCameraBody: "ক্যামেরা সংযুক্ত আছে কিনা এবং অন্য অ্যাপ ব্যবহার করছে কিনা দেখে আবার চেষ্টা করুন।",
    errBlank: "ক্যামেরা থেকে কোনো ছবি আসছে না",
    errBlankBody: "ভার্চুয়াল ক্যামেরা হলে ফোন বা উৎস সংযুক্ত করুন, অথবা অন্য ক্যামেরা বেছে নিন।",
    errImage: "ছবিটি বিশ্লেষণ করা যায়নি",
    errVideo: "ভিডিওটি প্রক্রিয়া করা যায়নি",
    errGeneric: "একটি সমস্যা হয়েছে",
    audioBlocked: "কণ্ঠস্বর চালু করতে একবার স্পর্শ করুন",
    srcDevice: "এই ডিভাইস", srcServer: "সার্ভার ক্যামেরা",
    switchCam: "ক্যামেরা বদলান",
    camAsking: "ক্যামেরার অনুমতি চাওয়া হচ্ছে…",
    camDenied: "ক্যামেরার অনুমতি দেওয়া হয়নি",
    camDeniedBody: "ব্রাউজারের সেটিংসে এই সাইটের জন্য ক্যামেরা চালু করে আবার স্টার্ট চাপুন।",
    camNone: "কোনো ক্যামেরা পাওয়া যায়নি",
    camNoneBody: "এই ডিভাইসে ব্যবহারযোগ্য ক্যামেরা নেই।",
    camInsecure: "ক্যামেরার জন্য নিরাপদ সংযোগ দরকার",
    camInsecureBody: "https:// ঠিকানা (আপনার Cloudflare Tunnel) দিয়ে খুলুন — সাধারণ http-তে ব্রাউজার ক্যামেরা বন্ধ রাখে।",
    camUnsupported: "এই ব্রাউজারে ক্যামেরা চালু করা যাচ্ছে না",
    camUnsupportedBody: "অ্যান্ড্রয়েডে Chrome অথবা আইফোনে Safari ব্যবহার করুন।",
    camBusy: "ক্যামেরা ব্যবহৃত হচ্ছে",
    camBusyBody: "অন্য অ্যাপ বা ট্যাব ক্যামেরা ব্যবহার করছে। সেটি বন্ধ করে আবার চেষ্টা করুন।",
    detecting: "বস্তু শনাক্ত করা হচ্ছে…",
    serverDown: "শনাক্তকরণ সার্ভার পাওয়া যাচ্ছে না",
    serverDownBody: "পেজ খুলেছে কিন্তু সার্ভার উত্তর দেয়নি। টানেল চালু আছে কিনা দেখুন।",
    placeholderModel: "সাধারণ মডেল চলছে, প্রশিক্ষিত থিসিস মডেল নয়।"
  }
};

/* Bengali display names, mirroring backend/voice/message_generator.py. */
const BN_CLASS = {
  person: "মানুষ", pole: "খুঁটি", tree: "গাছ", stairs: "সিঁড়ি",
  pothole: "গর্ত", open_drain: "খোলা ড্রেন", parked_vehicle: "পার্ক করা গাড়ি",
  street_obstacle: "রাস্তার প্রতিবন্ধকতা", curb: "ফুটপাথের কিনারা",
  chair: "চেয়ার", couch: "সোফা", bed: "বিছানা", "dining table": "টেবিল",
  tv: "টেলিভিশন", toilet: "টয়লেট", sink: "সিংক", refrigerator: "ফ্রিজ",
  oven: "ওভেন", microwave: "মাইক্রোওয়েভ", "potted plant": "গাছের টব",
  bench: "বেঞ্চ", rickshaw: "রিকশা", car: "গাড়ি", bus: "বাস",
  motorcycle: "মোটরসাইকেল", bicycle: "সাইকেল", truck: "ট্রাক"
};

/* Safety-relevant classes, matching DANGER_CLASSES in the navigation engine.
   Kept in step so the list order here cannot contradict what is spoken. */
const HAZARD = new Set(["pothole", "open_drain", "stairs", "curb", "pole",
                        "parked_vehicle", "car", "bus", "truck", "motorcycle",
                        "bicycle", "rickshaw", "train"]);

/* ----------------------------------------------------------------- state */

const state = {
  lang: "bn", mode: "live",
  running: false, detectionOn: true, indoorOn: true, voiceOn: true,
  liveCamera: null, videoLoop: false, announceFar: false,
  imageFile: null, videoFile: null,
  spoken: new Set(), failures: 0, timer: null, busy: false,
  source: "device", facing: "environment"
};

/* ------------------------------------------------------------------- dom */

const $ = (id) => document.getElementById(id);
const el = {};
[
  "srGuidance","srStatus","settingsBtn","errorNotice","errorTitle","errorText","retryBtn",
  "modelNotice","modelText","modeLive","modeImage","modeVideo","viewport","videoFeed",
  "stillImage","vpEmpty","vpTitle","vpHint","vpLoading","loadingText","zones","liveChip",
  "mObjects","mFps","mInference","mModel","playback","pbNow","pbTotal","pbFill","pbTrack",
  "pbPlay","pbPause","pbRestart","pbState","cameraActions","startBtn","stopBtn","stopAllBtn",
  "imagePanel","imageInput","imageDrop","imageName","analyseBtn","clearBtn",
  "videoPanel","videoInput","videoDrop","videoName","loadVideoBtn","loopChk",
  "safetyStatus","statusIcon","statusText","guidance","guideArrow","guideEn","guideBn",
  "dirLeft","dirAhead","dirRight","voiceToggle","voiceState","langBn","langEn",
  "detCount","objectList","objEmpty","sheetBackdrop","settingsPanel","closeSettings",
  "localVideo","overlay","sourceRow","srcDevice","srcServer","switchBtn",
  "detectionToggle","detState","indoorToggle","indoorState","confidence","confValue",
  "cooldown","cooldownValue","farToggle","farState","sheetMeta"
].forEach((id) => { el[id] = $(id); });

const t = (k) => (I18N[state.lang] && I18N[state.lang][k]) || I18N.en[k] || k;

function className(name) {
  if (state.lang === "bn") return BN_CLASS[name] || name.replace(/_/g, " ");
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* -------------------------------------------------------------------- api */

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

const jsonPost = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body)
});

/* ----------------------------------------------------------------- notice */

function showError(title, body) {
  const text = (body == null ? "" : String(body)).trim();
  if (!title && !text) { clearError(); return; }
  el.errorTitle.textContent = title || t("errGeneric");
  el.errorText.textContent = text;
  el.errorNotice.hidden = false;
  el.srStatus.textContent = (title || "") + ". " + text;
}
function clearError() {
  el.errorNotice.hidden = true;
  el.errorText.textContent = "";
}

/* ------------------------------------------------------------------ audio */
/* One persistent element, played in order. A local `new Audio()` can be
   garbage-collected mid-sentence, several clips fired at once talk over each
   other, and a blocked autoplay must be explained rather than swallowed. */

const audio = {
  el: null, queue: [], playing: false, unlocked: false, blocked: false,

  element() {
    if (!this.el) {
      this.el = document.createElement("audio");
      this.el.preload = "auto";
      this.el.addEventListener("ended", () => { this.playing = false; this.next(); });
      this.el.addEventListener("error", () => { this.playing = false; this.next(); });
      document.body.appendChild(this.el);
    }
    return this.el;
  },

  push(url) {
    this.queue.push(url);
    if (this.queue.length > 3) this.queue = this.queue.slice(-3);  // stale > missing
    this.next();
  },

  next() {
    if (this.playing || !this.queue.length) return;
    const url = this.queue.shift();
    const a = this.element();
    a.src = url;
    this.playing = true;
    a.play().then(() => {
      this.unlocked = true;
      if (this.blocked) { this.blocked = false; clearError(); }
    }).catch((err) => {
      this.playing = false;
      if (err && err.name === "NotAllowedError") {
        this.queue.unshift(url);
        if (!this.blocked) { this.blocked = true; showError(t("audioBlocked"), ""); }
      }
    });
  },

  stop() {
    this.queue = []; this.playing = false;
    if (this.el) { this.el.pause(); try { this.el.currentTime = 0; } catch (e) {} }
  }
};
document.addEventListener("click", () => { if (!audio.unlocked) audio.next(); }, true);

/* ------------------------------------------------------- device camera */
/*
   The phone's own camera.

   The server-camera path (/video_feed) opens a camera with OpenCV on the
   machine running the backend. Loaded on a phone that shows the LAPTOP's
   webcam, which is useless for walking around. So on a device with its own
   camera we take the stream here with getUserMedia and post frames to
   /api/detect_frame instead. The model, the spatial rules and the navigation
   engine are untouched and still run on the server.

   Three things this has to get right:

     * Single flight. Inference takes ~40-250 ms and the network adds more.
       Firing a request per animation frame would queue hundreds of them, so
       only one request is ever in the air and newer frames are dropped while
       it is out. A stale frame is worth nothing to someone walking.

     * Capture size. Frames are downscaled before encoding. A 1080p JPEG over
       a mobile connection costs far more latency than it buys accuracy at
       640px, which is the size the model runs at anyway.

     * Overlay scale. The video is letterboxed inside its box with
       object-fit: contain, so the displayed rectangle is NOT the element
       rectangle. Boxes come back normalised and are mapped through the
       letterbox, otherwise they drift on any aspect ratio but the camera's.
*/

const CAPTURE = {
  maxSide: 640,        // longest edge sent to the server
  quality: 0.6,        // JPEG quality: legible to the model, small on the wire
  minIntervalMs: 140   // ~7 fps ceiling; the single-flight lock does the rest
};

const deviceCam = {
  stream: null, running: false, inFlight: false, timer: null,
  canvas: null, ctx: null, abort: null,

  supported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  },

  async start() {
    if (!window.isSecureContext) {
      throw Object.assign(new Error("insecure"), { kind: "insecure" });
    }
    if (!this.supported()) {
      throw Object.assign(new Error("unsupported"), { kind: "unsupported" });
    }

    // Ask for the rear camera but do not insist: `ideal` lets a laptop with
    // only a front camera still succeed instead of throwing.
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: state.facing },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      // Some devices reject the facingMode hint outright; retry plainly before
      // giving up, so a plain desktop webcam is not refused.
      if (err && (err.name === "OverconstrainedError" || err.name === "NotFoundError")) {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } else {
        throw err;
      }
    }

    el.localVideo.srcObject = this.stream;
    el.localVideo.hidden = false;
    el.overlay.hidden = false;
    await el.localVideo.play().catch(() => {});

    // videoWidth stays 0 until the first frame lands, and every scaling
    // calculation below depends on it.
    await new Promise((res) => {
      if (el.localVideo.videoWidth) return res();
      el.localVideo.addEventListener("loadedmetadata", res, { once: true });
      setTimeout(res, 3000);
    });

    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.running = true;
    this.loop();
  },

  stop() {
    this.running = false;
    if (this.timer) { clearTimeout(this.timer); this.timer = null; }
    if (this.abort) { try { this.abort.abort(); } catch (e) {} this.abort = null; }
    if (this.stream) {
      // Stopping every track is what turns the camera indicator off.
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    el.localVideo.srcObject = null;
    el.localVideo.hidden = true;
    el.overlay.hidden = true;
    this.inFlight = false;
  },

  async switchCamera() {
    state.facing = state.facing === "environment" ? "user" : "environment";
    if (!this.running) return;
    this.stop();
    try {
      await this.start();
      Viewport.showDeviceCamera();
    } catch (err) {
      reportCameraError(err);
    }
  },

  loop() {
    if (!this.running) return;
    this.timer = setTimeout(() => {
      this.tick().finally(() => this.loop());
    }, CAPTURE.minIntervalMs);
  },

  async tick() {
    if (!this.running || this.inFlight) return;        // single flight
    const v = el.localVideo;
    if (!v.videoWidth || !v.videoHeight) return;

    const scale = Math.min(1, CAPTURE.maxSide / Math.max(v.videoWidth, v.videoHeight));
    this.canvas.width = Math.round(v.videoWidth * scale);
    this.canvas.height = Math.round(v.videoHeight * scale);
    this.ctx.drawImage(v, 0, 0, this.canvas.width, this.canvas.height);

    const blob = await new Promise((res) =>
      this.canvas.toBlob(res, "image/jpeg", CAPTURE.quality));
    if (!blob || !this.running) return;

    const fd = new FormData();
    fd.append("file", blob, "frame.jpg");
    fd.append("language", state.lang);
    fd.append("speak", String(state.voiceOn));

    this.inFlight = true;
    this.abort = new AbortController();
    try {
      const res = await fetch("/api/detect_frame", {
        method: "POST", body: fd, signal: this.abort.signal
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      if (!this.running) return;                       // stopped mid-request
      renderDeviceResult(data);
      state.failures = 0;
      clearError();
    } catch (err) {
      if (err.name === "AbortError") return;
      state.failures += 1;
      if (state.failures > 3) showError(t("serverDown"), t("serverDownBody"));
    } finally {
      this.inFlight = false;
      this.abort = null;
    }
  }
};

function reportCameraError(err) {
  const name = (err && (err.kind || err.name)) || "";
  const map = {
    insecure:              ["camInsecure", "camInsecureBody"],
    unsupported:           ["camUnsupported", "camUnsupportedBody"],
    NotAllowedError:       ["camDenied", "camDeniedBody"],
    PermissionDeniedError: ["camDenied", "camDeniedBody"],
    NotFoundError:         ["camNone", "camNoneBody"],
    DevicesNotFoundError:  ["camNone", "camNoneBody"],
    NotReadableError:      ["camBusy", "camBusyBody"],
    TrackStartError:       ["camBusy", "camBusyBody"]
  };
  const pair = map[name] || ["errCamera", "errCameraBody"];
  showError(t(pair[0]), t(pair[1]));
}

/* Draw the boxes the server sent over the live video.

   object-fit: contain letterboxes the frame inside the element, so the picture
   occupies only part of it. Mapping normalised boxes straight onto the element
   would misplace every one of them in portrait, which is the common case on a
   phone. */
function paintOverlay(detections) {
  const v = el.localVideo, c = el.overlay;
  if (!v.videoWidth) return;

  const box = v.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(box.width * dpr) || c.height !== Math.round(box.height * dpr)) {
    c.width = Math.round(box.width * dpr);
    c.height = Math.round(box.height * dpr);
    c.style.width = box.width + "px";
    c.style.height = box.height + "px";
  }

  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, box.width, box.height);

  // Where the picture actually sits inside the element.
  const vidAR = v.videoWidth / v.videoHeight;
  const boxAR = box.width / box.height;
  let drawW, drawH;
  if (vidAR > boxAR) { drawW = box.width; drawH = box.width / vidAR; }
  else               { drawH = box.height; drawW = box.height * vidAR; }
  const offX = (box.width - drawW) / 2;
  const offY = (box.height - drawH) / 2;

  ctx.lineWidth = 2;
  ctx.font = "600 13px Inter, system-ui, sans-serif";
  ctx.textBaseline = "top";

  (detections || []).forEach((d) => {
    const x1 = d.bbox_norm[0], y1 = d.bbox_norm[1];
    const x2 = d.bbox_norm[2], y2 = d.bbox_norm[3];
    const x = offX + x1 * drawW, y = offY + y1 * drawH;
    const w = (x2 - x1) * drawW, h = (y2 - y1) * drawH;
    const colour = HAZARD.has(d.class_name) ? "#dc2626" : "#1d4ed8";

    ctx.strokeStyle = colour;
    ctx.strokeRect(x, y, w, h);

    const label = className(d.class_name) + "  " + Math.round(d.confidence * 100) + "%";
    const tw = ctx.measureText(label).width + 10;
    const ty = y > 20 ? y - 19 : y + h + 1;
    ctx.fillStyle = colour;
    ctx.fillRect(x, ty, tw, 19);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 5, ty + 3);
  });
}

/* One frame's result -> the same panels the desktop already uses. */
function renderDeviceResult(d) {
  paintOverlay(d.detections);
  Objects.render(d.detections);
  Safety.render(d.safety, true);
  Guidance.render({
    running: true,
    guidance: d.guidance,
    urgent: d.urgent,
    detections: d.detections,
    detection_enabled: true
  });

  el.mInference.textContent = Math.round(d.inference_ms) + " ms";
  el.mFps.textContent = d.inference_ms
    ? (1000 / Math.max(d.inference_ms, CAPTURE.minIntervalMs)).toFixed(1) : "—";

  if (!state.voiceOn) return;

  // Speech comes back with the frame rather than through /api/state, which
  // reports the server pipeline and is silent in this mode.
  (d.play_audio || []).forEach((clip) => {
    if (state.spoken.has(clip.id)) return;
    state.spoken.add(clip.id);
    audio.push(clip.url);
  });
  if ("speechSynthesis" in window) {
    (d.speak_browser || []).forEach((phrase) => {
      if (state.spoken.has(phrase.id)) return;
      state.spoken.add(phrase.id);
      const utter = new SpeechSynthesisUtterance(phrase.text);
      utter.lang = phrase.lang;
      window.speechSynthesis.speak(utter);
    });
  }
  if (state.spoken.size > 200) state.spoken = new Set();
}

/* --------------------------------------------------------------- viewport */

const Viewport = {
  showDeviceCamera() {
    el.localVideo.hidden = false;
    el.overlay.hidden = false;
    this.detachStream();
    el.stillImage.hidden = true;
    el.vpEmpty.hidden = true; el.vpLoading.hidden = true;
    el.zones.hidden = false; el.liveChip.hidden = false;
  },

  showStream() {
    if (el.videoFeed.hidden) {
      el.videoFeed.src = "/video_feed?t=" + Date.now();
      el.videoFeed.hidden = false;
    }
    el.stillImage.hidden = true;
    el.vpEmpty.hidden = true; el.vpLoading.hidden = true;
    el.zones.hidden = false; el.liveChip.hidden = false;
  },
  showImage(url) {
    el.stillImage.src = url;
    el.stillImage.hidden = false;
    this.detachStream();
    el.vpEmpty.hidden = true; el.vpLoading.hidden = true;
    el.zones.hidden = false; el.liveChip.hidden = true;
  },
  showIdle(titleKey, hintKey) {
    this.detachStream();
    el.stillImage.hidden = true;
    el.localVideo.hidden = true;
    el.overlay.hidden = true;
    el.vpTitle.textContent = t(titleKey);
    el.vpHint.textContent = t(hintKey);
    el.vpEmpty.hidden = false; el.vpLoading.hidden = true;
    el.zones.hidden = true; el.liveChip.hidden = true;
  },
  showLoading(key) {
    el.loadingText.textContent = t(key);
    el.vpLoading.hidden = false;
  },
  hideLoading() { el.vpLoading.hidden = true; },
  detachStream() {
    if (!el.videoFeed.hidden) {
      el.videoFeed.hidden = true;
      el.videoFeed.removeAttribute("src");
    }
  }
};

/* ----------------------------------------------------------------- safety */

const SAFETY = {
  clear:     { icon: "✓", key: "safeClear" },
  attention: { icon: "•", key: "safeAttention" },
  warning:   { icon: "!", key: "safeWarning" },
  critical:  { icon: "!", key: "safeCritical" }
};

const Safety = {
  render(safety, active) {
    // With nothing running the system is not asserting that the path is clear
    // - it simply is not looking. Saying "clear" there would be a claim the
    // system has not earned, which in an assistive tool is worse than silence.
    if (active === false) {
      el.safetyStatus.dataset.level = "attention";
      el.statusIcon.textContent = "•";
      el.statusText.textContent = t("notRunning");
      return;
    }
    const level = (safety && safety.level) || "clear";
    const spec = SAFETY[level] || SAFETY.clear;
    if (el.safetyStatus.dataset.level !== level) {
      el.safetyStatus.dataset.level = level;
      el.srStatus.textContent = t(spec.key);
    }
    el.statusIcon.textContent = spec.icon;
    el.statusText.textContent = t(spec.key);
  }
};

/* --------------------------------------------------------------- guidance */

const Guidance = {
  render(data) {
    const detections = data.detections || [];
    const urgent = !!data.urgent;

    let en = "", bn = "";
    if (!data.running && state.mode === "live") {
      en = I18N.en.notRunning; bn = I18N.bn.notRunning;
    } else if (data.detection_enabled === false) {
      en = I18N.en.detectionOff; bn = I18N.bn.detectionOff;
    } else if (data.guidance) {
      // The backend generates in the ACTIVE language; the other line is a
      // short local rendering so both are always visible.
      if (state.lang === "bn") { bn = data.guidance; en = this.mirror(detections, "en"); }
      else                     { en = data.guidance; bn = this.mirror(detections, "bn"); }
    } else {
      en = I18N.en.pathClear; bn = I18N.bn.pathClear;
    }

    el.guideEn.textContent = en;
    el.guideBn.textContent = bn;

    const primary = state.lang === "bn" ? bn : en;
    if (this._last !== primary) {
      this._last = primary;
      el.srGuidance.textContent = primary;
    }

    // Direction from the highest-priority object, not from the sentence.
    const top = topObject(detections);
    const pos = top ? top.position : null;
    el.guideArrow.textContent = urgent ? "!" : pos === "LEFT" ? "←"
                              : pos === "RIGHT" ? "→" : "↑";
    el.guidance.classList.toggle("is-urgent", urgent);

    [["LEFT", el.dirLeft], ["CENTER", el.dirAhead], ["RIGHT", el.dirRight]]
      .forEach(([name, node]) => {
        const on = pos === name;
        node.classList.toggle("is-on", on && !urgent);
        node.classList.toggle("is-alert", on && urgent);
      });
  },

  /* A compact rendering of the top object in the non-active language. */
  mirror(detections, lang) {
    const top = topObject(detections);
    if (!top) return lang === "bn" ? I18N.bn.pathClear : I18N.en.pathClear;
    const name = lang === "bn"
      ? (BN_CLASS[top.class_name] || top.class_name.replace(/_/g, " "))
      : top.class_name.replace(/_/g, " ");
    if (lang === "bn") {
      const place = top.position === "LEFT" ? "বামে"
                  : top.position === "RIGHT" ? "ডানে" : "সামনে";
      return place + " " + name + " আছে";
    }
    const place = top.position === "LEFT" ? "on your left"
                : top.position === "RIGHT" ? "on your right" : "ahead";
    return name.charAt(0).toUpperCase() + name.slice(1) + " " + place;
  }
};

/* Rank detections the same way the navigation engine does, so the list order
   and the arrow agree with what is spoken. */
function priority(d) {
  let s = 0;
  if (HAZARD.has(d.class_name)) s += 100;
  if (d.position === "CENTER") s += 50;
  s += d.proximity === "NEAR" ? 30 : d.proximity === "MEDIUM" ? 12 : 0;
  s += (d.confidence || 0) * 10;
  return s;
}
/* Mirror NavigationEngine._relevant(): distant non-hazards are not announced
   unless the user asks for them. Without this the arrow and the English mirror
   could point at an object the spoken sentence never mentions. */
function relevant(detections) {
  return (detections || []).filter((d) =>
    state.announceFar || d.proximity !== "FAR" || HAZARD.has(d.class_name));
}

function topObject(detections) {
  const pool = relevant(detections);
  if (!pool.length) return null;
  return pool.slice().sort((a, b) => priority(b) - priority(a))[0];
}

/* ---------------------------------------------------------------- objects */

const Objects = {
  render(detections) {
    // Announceable objects first, then the rest. Sorting purely by raw score
    // would put a distant object above the one being spoken about, so the
    // top of the list would disagree with the guidance card.
    const all = detections || [];
    const keep = new Set(relevant(all));
    const list = all.slice().sort((a, b) => {
      const ra = keep.has(a) ? 1 : 0, rb = keep.has(b) ? 1 : 0;
      return rb - ra || priority(b) - priority(a);
    });
    el.detCount.textContent = String(list.length);
    el.mObjects.textContent = String(list.length);

    if (!list.length) {
      el.objectList.innerHTML =
        '<li class="obj-empty">' + esc(t("noObjects")) + "</li>";
      return;
    }

    el.objectList.innerHTML = list.map((d) => {
      const hazard = HAZARD.has(d.class_name);
      const arrow = d.position === "LEFT" ? "←" : d.position === "RIGHT" ? "→" : "↑";
      const where = t((d.position || "CENTER") === "LEFT" ? "left"
                    : d.position === "RIGHT" ? "right" : "ahead");
      const prox = t((d.proximity || "MEDIUM") === "NEAR" ? "near"
                   : d.proximity === "FAR" ? "far" : "medium");
      // Hazard is marked by an icon AND a word, never by colour alone.
      const flag = hazard ? ' <span aria-hidden="true">•</span> ⚠' : "";
      return '<li class="obj' + (hazard ? " is-hazard" : "") + '">' +
               '<span class="obj-dir" aria-hidden="true">' + arrow + "</span>" +
               '<span class="obj-body">' +
                 '<span class="obj-name' + (state.lang === "bn" ? " bn" : "") + '">' +
                   esc(className(d.class_name)) + "</span>" +
                 '<span class="obj-meta">' + esc(where) + " · " + esc(prox) + flag + "</span>" +
               "</span>" +
               '<span class="obj-conf">' + Math.round((d.confidence || 0) * 100) + "%</span>" +
             "</li>";
    }).join("");
  }
};

/* --------------------------------------------------------------- playback */

function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return m + ":" + String(s).padStart(2, "0");
}

const Playback = {
  render(pb) {
    if (!pb || !pb.is_file) return;
    const pct = pb.percent || 0;
    el.pbFill.style.width = pct + "%";
    el.pbTrack.setAttribute("aria-valuenow", String(Math.round(pct)));
    el.pbNow.textContent = fmtTime(pb.position_s);
    el.pbTotal.textContent = fmtTime(pb.duration_s);
    el.pbState.textContent = pb.finished ? t("finished")
                           : pb.paused ? t("paused") : t("playing");
    el.pbPlay.disabled = !pb.paused && !pb.finished;
    el.pbPause.disabled = pb.paused || pb.finished;
  },
  async cmd(action) {
    try { this.render((await jsonPost("/api/playback", { action })).playback); }
    catch (err) { showError(t("errVideo"), err.message); }
  }
};

/* ------------------------------------------------------------------- mode */

const Mode = {
  apply() {
    const m = state.mode;
    [["live", el.modeLive], ["image", el.modeImage], ["video", el.modeVideo]]
      .forEach(([name, btn]) => {
        const on = name === m;
        btn.classList.toggle("is-active", on);
        btn.setAttribute("aria-selected", String(on));
      });

    el.cameraActions.hidden = m !== "live";
    el.imagePanel.hidden = m !== "image";
    el.videoPanel.hidden = m !== "video";
    el.playback.hidden = m !== "video";
    el.metrics.hidden = false;

    if (m === "image") Viewport.showIdle("imageIdle", "imageIdleHint");
    else if (m === "video") Viewport.showIdle("videoIdle", "videoIdleHint");
    else if (!state.running) Viewport.showIdle("cameraOff", "cameraOffHint");
  },

  async set(mode) {
    if (mode === state.mode) return;
    try {
      const body = { mode };
      if (mode === "live" && state.liveCamera) body.camera = state.liveCamera;
      const d = await jsonPost("/api/mode", body);
      state.mode = mode;
      if (d.camera_source) state.liveCamera = d.camera_source;
      state.running = false;
      clearError();
      this.apply();
    } catch (err) { showError(t("errGeneric"), err.message); }
  }
};

/* --------------------------------------------------------------- controls */

async function startCamera() {
  if (state.busy) return;
  state.busy = true;
  el.startBtn.disabled = true;
  clearError();

  // "This device" takes the camera in the browser; "Server camera" keeps the
  // original OpenCV path. Both feed the same model.
  if (state.source === "device") {
    Viewport.showLoading("camAsking");
    try {
      // Speech is unlocked here, inside the click that started the camera:
      // mobile browsers only allow audio to begin from a user gesture.
      unlockSpeech();
      await deviceCam.start();
      state.running = true;
      Viewport.showDeviceCamera();
    } catch (err) {
      deviceCam.stop();
      Viewport.hideLoading();
      Viewport.showIdle("cameraOff", "cameraOffHint");
      reportCameraError(err);
    } finally {
      state.busy = false;
      el.startBtn.disabled = false;
    }
    return;
  }

  Viewport.showLoading("starting");
  try {
    await api("/api/start", { method: "POST" });
    state.running = true;
    Viewport.showStream();
  } catch (err) {
    Viewport.hideLoading();
    Viewport.showIdle("cameraOff", "cameraOffHint");
    showError(t("errCamera"), err.message || t("errCameraBody"));
  } finally {
    state.busy = false;
    el.startBtn.disabled = false;
  }
}

/* Mobile browsers refuse speech that did not originate in a user gesture.
   Speaking one empty utterance inside the Start click opens the door for
   every later announcement. */
function unlockSpeech() {
  if (!("speechSynthesis" in window)) return;
  try {
    const u = new SpeechSynthesisUtterance(" ");
    u.volume = 0;
    window.speechSynthesis.speak(u);
  } catch (e) { /* not fatal: server-side audio still works */ }
}

async function stopCamera() {
  // Release the phone's camera first so the indicator goes out immediately,
  // even if the server call is slow.
  deviceCam.stop();
  try { await api("/api/stop", { method: "POST" }); } catch (e) { /* already stopped */ }
  state.running = false;
  audio.stop();
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  Viewport.showIdle(state.mode === "video" ? "videoIdle" : "cameraOff",
                    state.mode === "video" ? "videoIdleHint" : "cameraOffHint");
  el.mFps.textContent = "—"; el.mInference.textContent = "—";
}

async function stopEverything() {
  audio.stop();
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  try { await jsonPost("/api/voice", { enabled: false }); } catch (e) {}
  setVoice(false);
  await stopCamera();
}

function setVoice(on) {
  state.voiceOn = on;
  el.voiceToggle.classList.toggle("is-off", !on);
  el.voiceToggle.setAttribute("aria-checked", String(on));
  el.voiceState.textContent = on ? t("on") : t("off");
  if (!on) audio.stop();
}

async function toggleVoice() {
  const next = !state.voiceOn;
  try { await jsonPost("/api/voice", { enabled: next }); setVoice(next); }
  catch (err) { showError(t("errGeneric"), err.message); }
}

async function setLanguage(lang) {
  if (lang === state.lang) return;
  try {
    await jsonPost("/api/language", { language: lang });
    state.lang = lang;
    applyLanguage();
  } catch (err) { showError(t("errGeneric"), err.message); }
}

function applyLanguage() {
  document.documentElement.lang = state.lang;
  document.body.classList.toggle("lang-bn", state.lang === "bn");
  el.langBn.classList.toggle("is-active", state.lang === "bn");
  el.langEn.classList.toggle("is-active", state.lang === "en");
  el.langBn.setAttribute("aria-pressed", String(state.lang === "bn"));
  el.langEn.setAttribute("aria-pressed", String(state.lang === "en"));

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const k = node.dataset.i18n;
    if (I18N[state.lang] && I18N[state.lang][k]) node.textContent = I18N[state.lang][k];
  });
  el.voiceState.textContent = state.voiceOn ? t("on") : t("off");
  Mode.apply();
}

/* --------------------------------------------------------------- settings */

function pill(node, label, on) {
  node.classList.toggle("is-off", !on);
  node.setAttribute("aria-checked", String(on));
  label.textContent = on ? t("on") : t("off");
}

const Settings = {
  open() {
    el.settingsPanel.hidden = false;
    el.sheetBackdrop.hidden = false;
    el.settingsBtn.setAttribute("aria-expanded", "true");
    el.closeSettings.focus();
  },
  close() {
    el.settingsPanel.hidden = true;
    el.sheetBackdrop.hidden = true;
    el.settingsBtn.setAttribute("aria-expanded", "false");
    el.settingsBtn.focus();
  },
  async patch(body) {
    try { await jsonPost("/api/settings", body); }
    catch (err) { showError(t("errGeneric"), err.message); }
  }
};

/* ------------------------------------------------------------------- poll */

async function poll() {
  try {
    const data = await api("/api/state");
    state.failures = 0;

    if (!data.ok) { showError(t("errGeneric"), data.error || ""); return; }

    state.running = Boolean(data.running);
    if (typeof data.detection_enabled === "boolean") {
      state.detectionOn = data.detection_enabled;
      pill(el.detectionToggle, el.detState, state.detectionOn);
    }

    // Errors that the backend actually reports.
    if (data.camera_error) showError(t("errCamera"), data.camera_error);
    else if (data.camera_blank) showError(t("errBlank"), t("errBlankBody"));
    else if (!audio.blocked) clearError();

    // In device-camera mode the frame responses own these panels; the poll
    // reports the SERVER pipeline, which is idle, and would blank them.
    if (state.mode === "live" && state.source === "device" && deviceCam.running) {
      return;
    }

    if (state.mode !== "image") {
      el.mFps.textContent = data.fps ? Number(data.fps).toFixed(1) : "—";
      el.mInference.textContent = data.inference_ms
        ? Math.round(data.inference_ms) + " ms" : "—";
    }

    // Image mode is a one-shot analysis, not a stream. The live pipeline is
    // idle there, so its (empty) detections would immediately overwrite the
    // analysis result the user just asked for. Metrics still update, because
    // those describe the backend rather than the image.
    if (state.mode !== "image") {
      const looking = state.running && data.detection_enabled !== false;
      Safety.render(data.safety, looking);
      Guidance.render(data);
      Objects.render(data.detections);
      if (state.mode === "video") Playback.render(data.playback);
    }

    // Keep the viewport in step with the BACKEND, not with which button was
    // pressed in this tab — covers reloads and a second open tab.
    if (state.mode !== "image") {
      if (state.running) Viewport.showStream();
      else if (!el.videoFeed.hidden) {
        Viewport.showIdle(state.mode === "video" ? "videoIdle" : "cameraOff",
                          state.mode === "video" ? "videoIdleHint" : "cameraOffHint");
      }
    }

    if (state.voiceOn) {
      (data.play_audio || []).forEach((c) => {
        if (state.spoken.has(c.id)) return;
        state.spoken.add(c.id); audio.push(c.url);
      });
      if ("speechSynthesis" in window) {
        (data.speak_browser || []).forEach((p) => {
          if (state.spoken.has(p.id)) return;
          state.spoken.add(p.id);
          const u = new SpeechSynthesisUtterance(p.text);
          u.lang = p.lang; window.speechSynthesis.speak(u);
        });
      }
      if (state.spoken.size > 200) state.spoken = new Set();
    }
  } catch (err) {
    state.failures += 1;
    if (state.failures > 3) showError(t("errGeneric"), err.message);
  }
}

/* ------------------------------------------------------------------- init */

async function loadInitialState() {
  try {
    const status = await api("/api/status");
    if (!status.ok) { showError(t("errGeneric"), status.error || ""); return; }

    const s = status.settings || {};
    state.lang = s.language || "bn";
    state.voiceOn = s.voice_enabled !== false;
    state.detectionOn = s.detection_enabled !== false;
    state.indoorOn = s.indoor_enabled !== false;
    state.mode = s.input_mode || "live";
    state.videoLoop = !!s.video_loop;
    if (state.mode === "live") state.liveCamera = s.camera_source;

    setVoice(state.voiceOn);
    pill(el.detectionToggle, el.detState, state.detectionOn);
    pill(el.indoorToggle, el.indoorState, state.indoorOn);
    state.announceFar = !!s.announce_far;
    pill(el.farToggle, el.farState, state.announceFar);
    el.loopChk.checked = state.videoLoop;

    el.confidence.value = String(Math.round((s.confidence || 0.3) * 100));
    el.confValue.value = el.confidence.value + "%";
    el.cooldown.value = String(s.announce_cooldown || 3);
    el.cooldownValue.value = Number(el.cooldown.value).toFixed(1) + "s";

    const model = status.model || {};
    el.mModel.textContent = model.path ? model.path.split(/[\\/]/).pop() : "—";
    el.sheetMeta.textContent =
      (model.device ? model.device.toUpperCase() + " · " : "") +
      (model.num_classes ? model.num_classes + " classes" : "");

    if (model.is_placeholder) {
      el.modelText.textContent = t("placeholderModel");
      el.modelNotice.hidden = false;
    }

    // Prefer this device's camera whenever the browser can open one. That is
    // the right default on a phone, and on a desktop it simply uses the
    // built-in webcam instead of the server's - both reach the same model.
    // Feature detection, not user-agent sniffing.
    const canUseDeviceCam = deviceCam.supported() && window.isSecureContext;
    state.source = canUseDeviceCam ? "device" : "server";
    el.srcDevice.classList.toggle("is-active", canUseDeviceCam);
    el.srcServer.classList.toggle("is-active", !canUseDeviceCam);
    el.srcDevice.setAttribute("aria-selected", String(canUseDeviceCam));
    el.srcServer.setAttribute("aria-selected", String(!canUseDeviceCam));
    el.srcDevice.disabled = !canUseDeviceCam;
    el.switchBtn.hidden = !canUseDeviceCam;
    if (!canUseDeviceCam && !window.isSecureContext) {
      el.srcDevice.title = I18N.en.camInsecureBody;
    }

    applyLanguage();
  } catch (err) {
    showError(t("errGeneric"), err.message);
  }
}

function bindEvents() {
  el.modeLive.addEventListener("click", () => Mode.set("live"));
  el.modeImage.addEventListener("click", () => Mode.set("image"));
  el.modeVideo.addEventListener("click", () => Mode.set("video"));

  // Source: this device's camera vs the server's. Default is chosen at boot
  // from whether this browser can actually open a camera.
  const pickSource = (src) => {
    if (src === state.source) return;
    if (state.running) stopCamera();
    state.source = src;
    el.srcDevice.classList.toggle("is-active", src === "device");
    el.srcServer.classList.toggle("is-active", src === "server");
    el.srcDevice.setAttribute("aria-selected", String(src === "device"));
    el.srcServer.setAttribute("aria-selected", String(src === "server"));
    el.switchBtn.hidden = src !== "device";
    clearError();
  };
  el.srcDevice.addEventListener("click", () => pickSource("device"));
  el.srcServer.addEventListener("click", () => pickSource("server"));
  el.switchBtn.addEventListener("click", () => deviceCam.switchCamera());

  // Rotating the phone changes the element box, so the overlay must be
  // re-measured or the boxes sit in the wrong place until the next frame.
  const remeasure = () => { if (deviceCam.running) paintOverlay([]); };
  window.addEventListener("orientationchange", remeasure);
  window.addEventListener("resize", remeasure);

  el.startBtn.addEventListener("click", startCamera);
  el.stopBtn.addEventListener("click", stopCamera);
  el.stopAllBtn.addEventListener("click", stopEverything);
  el.retryBtn.addEventListener("click", () => { clearError(); startCamera(); });

  el.voiceToggle.addEventListener("click", toggleVoice);
  el.langBn.addEventListener("click", () => setLanguage("bn"));
  el.langEn.addEventListener("click", () => setLanguage("en"));

  el.pbPlay.addEventListener("click", () => Playback.cmd("play"));
  el.pbPause.addEventListener("click", () => Playback.cmd("pause"));
  el.pbRestart.addEventListener("click", () => Playback.cmd("restart"));

  /* ---- image ---- */
  bindDropzone(el.imageDrop, el.imageInput, (f) => {
    state.imageFile = f;
    el.imageName.textContent = f.name + " · " + (f.size / 1e6).toFixed(1) + " MB";
    el.imageName.hidden = false;
    el.analyseBtn.disabled = false;
    Viewport.showImage(URL.createObjectURL(f));
  }, /^image\//);

  el.analyseBtn.addEventListener("click", analyseImage);
  el.clearBtn.addEventListener("click", () => {
    state.imageFile = null; el.imageInput.value = "";
    el.imageName.hidden = true; el.analyseBtn.disabled = true;
    Objects.render([]); clearError();
    Viewport.showIdle("imageIdle", "imageIdleHint");
  });

  /* ---- video ---- */
  bindDropzone(el.videoDrop, el.videoInput, (f) => {
    state.videoFile = f;
    el.videoName.textContent = f.name + " · " + (f.size / 1e6).toFixed(1) + " MB";
    el.videoName.hidden = false;
    el.loadVideoBtn.disabled = false;
  }, /^video\//);

  el.loadVideoBtn.addEventListener("click", loadVideo);
  el.loopChk.addEventListener("change", () => { state.videoLoop = el.loopChk.checked; });

  /* ---- settings ---- */
  el.settingsBtn.addEventListener("click", () => Settings.open());
  el.closeSettings.addEventListener("click", () => Settings.close());
  el.sheetBackdrop.addEventListener("click", () => Settings.close());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el.settingsPanel.hidden) Settings.close();
  });

  el.detectionToggle.addEventListener("click", () => {
    state.detectionOn = !state.detectionOn;
    pill(el.detectionToggle, el.detState, state.detectionOn);
    Settings.patch({ detection_enabled: state.detectionOn });
  });
  el.indoorToggle.addEventListener("click", () => {
    state.indoorOn = !state.indoorOn;
    pill(el.indoorToggle, el.indoorState, state.indoorOn);
    Settings.patch({ indoor_enabled: state.indoorOn });
  });
  el.farToggle.addEventListener("click", () => {
    const on = el.farToggle.classList.contains("is-off");
    state.announceFar = on;
    pill(el.farToggle, el.farState, on);
    Settings.patch({ announce_far: on });
  });

  el.confidence.addEventListener("input", () => {
    el.confValue.value = el.confidence.value + "%";
  });
  el.confidence.addEventListener("change", () => {
    Settings.patch({ confidence: Number(el.confidence.value) / 100 });
  });
  el.cooldown.addEventListener("input", () => {
    el.cooldownValue.value = Number(el.cooldown.value).toFixed(1) + "s";
  });
  el.cooldown.addEventListener("change", () => {
    Settings.patch({ announce_cooldown: Number(el.cooldown.value) });
  });

  el.videoFeed.addEventListener("error", () => {
    if (state.running && el.videoFeed.getAttribute("src")) {
      showError(t("errCamera"), t("errCameraBody"));
    }
  });
}

/* A dropzone that is also a real button: click, keyboard and drop all work. */
function bindDropzone(zone, input, onFile, accept) {
  const pick = () => input.click();
  zone.addEventListener("click", pick);
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
  });
  input.addEventListener("change", () => {
    const f = input.files && input.files[0];
    if (f) onFile(f);
  });
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault(); zone.classList.add("is-over");
  }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault(); zone.classList.remove("is-over");
  }));
  zone.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && accept.test(f.type)) onFile(f);
  });
}

async function analyseImage() {
  if (!state.imageFile) return;
  el.analyseBtn.disabled = true;
  Viewport.showLoading("analysing");
  clearError();

  const fd = new FormData();
  fd.append("file", state.imageFile);
  fd.append("language", state.lang);
  fd.append("speak", String(state.voiceOn));

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) {}
      throw new Error(detail);
    }
    const d = await res.json();
    Viewport.showImage(d.annotated_url + "?t=" + Date.now());
    Objects.render(d.detections);
    el.mInference.textContent = Math.round(d.inference_ms) + " ms";
    el.mFps.textContent = "—";
    Guidance.render({ running: true, guidance: d.guidance, urgent: d.urgent,
                      detections: d.detections, detection_enabled: true });
    Safety.render(d.detections && d.detections.length
      ? { level: d.urgent ? "critical" : "attention" } : { level: "clear" }, true);
  } catch (err) {
    Viewport.hideLoading();
    showError(t("errImage"), err.message);
  } finally {
    el.analyseBtn.disabled = false;
  }
}

async function loadVideo() {
  if (!state.videoFile) return;
  el.loadVideoBtn.disabled = true;
  Viewport.showLoading("loadingVideo");
  clearError();

  const fd = new FormData();
  fd.append("file", state.videoFile);
  fd.append("loop", String(state.videoLoop));
  fd.append("autostart", "true");

  try {
    const res = await fetch("/api/upload_video", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) {}
      throw new Error(detail);
    }
    const d = await res.json();
    state.running = !!d.running;
    Viewport.showStream();
    Playback.render(d.playback);
  } catch (err) {
    Viewport.hideLoading();
    Viewport.showIdle("videoIdle", "videoIdleHint");
    showError(t("errVideo"), err.message);
  } finally {
    el.loadVideoBtn.disabled = false;
  }
}

(function init() {
  bindEvents();
  loadInitialState().then(() => {
    poll();
    state.timer = setInterval(poll, POLL_MS);
  });
})();
