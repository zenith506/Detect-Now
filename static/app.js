// ---------------------------------------------------------------------------
// Where the backend lives.
//  - Page opened from localhost / 127.0.0.1: use a backend on the same computer (port 5000).
//  - Page opened from anywhere else (the hosted site): use the backend hosted on Render.
// To point at another server, add this line BEFORE app.js in index.html:
//    <script>window.DETECT_NOW_API_URL = "https://your-server/predict";</script>
// ---------------------------------------------------------------------------
const LOCAL_HOSTNAMES = ["localhost", "127.0.0.1"];
const DEFAULT_API_URL = "https://detect-now-backend.onrender.com/predict";

const API_URL =
    window.DETECT_NOW_API_URL ||
    (LOCAL_HOSTNAMES.includes(window.location.hostname)
        ? `${window.location.protocol}//${window.location.hostname}:5000/predict`
        : DEFAULT_API_URL);

const IMAGE_MAXIMUM_SIZE = 10 * 1024 * 1024; // must match the backend limit
const VIDEO_MAXIMUM_SIZE = 50 * 1024 * 1024;
const HISTORY_KEY = "detectNowHistory";

const mediaSelection = document.getElementById("media-selection");
const uploadArea = document.getElementById("upload-area");
const selectImageButton = document.getElementById("select-image-button");
const selectVideoButton = document.getElementById("select-video-button");
const switchToImageButton = document.getElementById("switch-to-image-button");
const backButton = document.getElementById("back-button");
const selectedTypeIcon = document.getElementById("selected-type-icon");
const uploadHeading = document.getElementById("upload-heading");
const uploadDescription = document.getElementById("upload-description");
const uploadText = document.getElementById("upload-text");
const fileInformation = document.getElementById("file-information");
const mediaInput = document.getElementById("media-input");
const imagePreviewSection = document.getElementById("image-preview-section");
const imagePreview = document.getElementById("image-preview");
const imageFileName = document.getElementById("image-file-name");
const videoPreviewSection = document.getElementById("video-preview-section");
const videoPreview = document.getElementById("video-preview");
const videoFileName = document.getElementById("video-file-name");
const videoNotice = document.getElementById("video-notice");
const analysisLoading = document.getElementById("analysis-loading");
const analysisError = document.getElementById("analysis-error");
const analysisErrorText = document.getElementById("analysis-error-text");
const detectionResult = document.getElementById("detection-result");
const predictionText = document.getElementById("prediction-text");
const confidenceText = document.getElementById("confidence-text");
const confidenceBarFill = document.getElementById("confidence-bar-fill");
const realProbability = document.getElementById("real-probability");
const fakeProbability = document.getElementById("fake-probability");
const analyseButton = document.getElementById("analyse-button");
const downloadReportButton = document.getElementById("download-report-button");
const uploadAgainButton = document.getElementById("upload-again-button");
const clearHistoryButton = document.getElementById("clear-history-button");
const historyList = document.getElementById("history-list");

let selectedMediaType = "";
let selectedFile = null;
let currentPreviewURL = "";
let latestResult = null;
let analysisRunId = 0; // lets us ignore a result that arrives after the user changed file


function openUploadArea(mediaType) {
    selectedMediaType = mediaType;
    resetSelectedFile();

    mediaSelection.hidden = true;
    uploadArea.hidden = false;

    if (mediaType === "image") {
        selectedTypeIcon.textContent = "▧";
        uploadHeading.textContent = "Upload Image";
        uploadDescription.textContent = "Select an image from your computer or device.";
        uploadText.textContent = "Choose an image";
        fileInformation.textContent = "JPG, JPEG, PNG, WEBP or BMP · Maximum 10 MB";
        mediaInput.accept =
            ".jpg,.jpeg,.png,.webp,.bmp,image/jpeg,image/png,image/webp,image/bmp";
    } else {
        selectedTypeIcon.textContent = "▶";
        uploadHeading.textContent = "Upload Video";
        uploadDescription.textContent = "Select a video from your computer or device.";
        uploadText.textContent = "Choose a video";
        fileInformation.textContent = "MP4, WebM or MOV · Maximum 50 MB";
        mediaInput.accept = ".mp4,.webm,.mov,video/mp4,video/webm,video/quicktime";
    }
}


function clearFaceBox() {
    // the preview may currently show a copy of the photo with the box painted on it
    if (imagePreview.dataset.boxed === "1" && currentPreviewURL) {
        imagePreview.src = currentPreviewURL;
    }
    delete imagePreview.dataset.boxed;

    const legacy = document.getElementById("face-box-canvas");
    if (legacy) {
        legacy.remove();
    }
}


function resetSelectedFile() {
    analysisRunId += 1;
    selectedFile = null;
    latestResult = null;
    mediaInput.value = "";

    imagePreviewSection.hidden = true;
    videoPreviewSection.hidden = true;
    videoNotice.hidden = true;
    analysisLoading.hidden = true;
    analysisError.hidden = true;
    detectionResult.hidden = true;
    analyseButton.hidden = true;
    downloadReportButton.hidden = true;
    uploadAgainButton.hidden = true;

    analyseButton.disabled = false;
    analyseButton.textContent = "Analyse Image";

    clearFaceBox();
    imagePreview.removeAttribute("src");
    imageFileName.textContent = "";

    videoPreview.pause();
    videoPreview.removeAttribute("src");
    videoPreview.load();
    videoFileName.textContent = "";

    confidenceBarFill.style.width = "0%";
    detectionResult.classList.remove("real-result", "fake-result", "uncertain-result");

    if (currentPreviewURL) {
        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = "";
    }
}


function resetResultAreas() {
    analysisRunId += 1;
    latestResult = null;

    analysisLoading.hidden = true;
    analysisError.hidden = true;
    detectionResult.hidden = true;
    videoNotice.hidden = true;
    downloadReportButton.hidden = true;

    clearFaceBox();
    confidenceBarFill.style.width = "0%";
    detectionResult.classList.remove("real-result", "fake-result", "uncertain-result");
}


function showError(message) {
    latestResult = null;

    analysisErrorText.textContent = message;
    analysisError.hidden = false;
    analysisLoading.hidden = true;
    detectionResult.hidden = true;
    downloadReportButton.hidden = true;

    clearFaceBox();
    confidenceBarFill.style.width = "0%";
}


selectImageButton.addEventListener("click", function () {
    openUploadArea("image");
});

selectVideoButton.addEventListener("click", function () {
    openUploadArea("video");
});

if (switchToImageButton) {
    switchToImageButton.addEventListener("click", function () {
        openUploadArea("image");
    });
}

backButton.addEventListener("click", function () {
    resetSelectedFile();
    uploadArea.hidden = true;
    mediaSelection.hidden = false;
    selectedMediaType = "";
});


mediaInput.addEventListener("change", function () {
    const file = mediaInput.files[0];

    if (!file) {
        return;
    }

    resetResultAreas();

    if (selectedMediaType === "image" && !file.type.startsWith("image/")) {
        showError("Please select a valid JPG, JPEG, PNG, WEBP or BMP image.");
        mediaInput.value = "";
        return;
    }

    if (selectedMediaType === "video" && !file.type.startsWith("video/")) {
        showError("Please select a valid MP4, WebM or MOV video.");
        mediaInput.value = "";
        return;
    }

    if (selectedMediaType === "image" && file.size > IMAGE_MAXIMUM_SIZE) {
        showError("Please select an image smaller than 10 MB.");
        mediaInput.value = "";
        return;
    }

    if (selectedMediaType === "video" && file.size > VIDEO_MAXIMUM_SIZE) {
        showError("Please select a video smaller than 50 MB.");
        mediaInput.value = "";
        return;
    }

    selectedFile = file;

    if (currentPreviewURL) {
        URL.revokeObjectURL(currentPreviewURL);
    }

    currentPreviewURL = URL.createObjectURL(file);

    if (selectedMediaType === "image") {
        imagePreview.src = currentPreviewURL;
        imageFileName.textContent = file.name;

        imagePreviewSection.hidden = false;
        videoPreviewSection.hidden = true;
        videoNotice.hidden = true;
        analyseButton.hidden = false;
    } else {
        videoPreview.src = currentPreviewURL;
        videoFileName.textContent = file.name;

        videoPreviewSection.hidden = false;
        imagePreviewSection.hidden = true;
        videoNotice.hidden = false;
        analyseButton.hidden = true;
    }

    uploadAgainButton.hidden = false;

    saveHistoryItem(file.name, selectedMediaType, file.size);
});


analyseButton.addEventListener("click", async function () {
    if (!selectedFile) {
        showError("Please select an image before starting analysis.");
        return;
    }

    resetResultAreas();
    const runId = analysisRunId;

    analysisLoading.hidden = false;
    analyseButton.disabled = true;
    analyseButton.textContent = "Analysing...";

    const formData = new FormData();
    formData.append("image", selectedFile, selectedFile.name);

    try {
        const response = await fetch(API_URL, { method: "POST", body: formData });

        let result;
        try {
            result = await response.json();
        } catch (error) {
            throw new Error("The detection server returned an invalid response.");
        }

        // The user picked another file while we were waiting: drop this result.
        if (runId !== analysisRunId) {
            return;
        }

        if (!response.ok || result.success === false) {
            throw new Error(
                result.message || result.error || "The image could not be analysed."
            );
        }

        latestResult = { ...result, analysedAt: new Date().toISOString() };
        displayResult(latestResult);
    } catch (error) {
        if (runId !== analysisRunId) {
            return;
        }

        if (error.message === "Failed to fetch" || error instanceof TypeError) {
            showError(
                "The detection server is unavailable. Start backend.py and try again."
            );
        } else {
            showError(error.message);
        }
    } finally {
        if (runId === analysisRunId) {
            analysisLoading.hidden = true;
            analyseButton.disabled = false;
            analyseButton.textContent = "Analyse Image Again";
        }
    }
});


const VERDICTS = {
    LIKELY_REAL: { label: "LIKELY REAL", cls: "real-result", report: "real" },
    UNCERTAIN: { label: "UNCERTAIN", cls: "uncertain-result", report: "uncertain" },
    LIKELY_DEEPFAKE: { label: "LIKELY DEEPFAKE", cls: "fake-result", report: "fake" },
    DEEPFAKE: { label: "DEEPFAKE", cls: "fake-result", report: "deepfake" }
};


function verdictInfo(result) {
    const known = VERDICTS[String(result.verdict || "")];
    if (known) {
        return known;
    }
    // older backend without bands: fall back to the two-way answer
    return String(result.prediction).toUpperCase() === "REAL"
        ? { label: "REAL", cls: "real-result", report: "real" }
        : { label: "DEEPFAKE", cls: "fake-result", report: "fake" };
}


function bandText(result) {
    const a = Number(result.band_real_below);
    const b = Number(result.band_uncertain_below);
    const c = Number(result.band_deepfake_from);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
        return "";
    }
    return "Below " + a + "% likely real · " + a + "% to under " + b + "% uncertain · " +
        b + "% to under " + c + "% likely deepfake · " + c + "% and above deepfake";
}


const RESULT_WARNING_ELEMENT = document.querySelector(".result-warning");
const RESULT_WARNING_TEXT = RESULT_WARNING_ELEMENT
    ? RESULT_WARNING_ELEMENT.textContent.trim()
    : "";


function displayResult(result) {
    const verdict = verdictInfo(result);
    const fakeScore = Number(result.fake_probability);
    const realScore = Number(result.real_probability);

    predictionText.textContent = verdict.label;
    confidenceText.textContent = fakeScore.toFixed(2) + "% deepfake score";
    realProbability.textContent = realScore.toFixed(2) + "%";
    fakeProbability.textContent = fakeScore.toFixed(2) + "%";

    confidenceBarFill.style.width = Math.min(100, Math.max(0, fakeScore)) + "%";

    detectionResult.classList.remove("real-result", "fake-result", "uncertain-result");
    detectionResult.classList.add(verdict.cls);

    detectionResult.hidden = false;
    analysisError.hidden = true;
    downloadReportButton.hidden = false;

    // Only the largest face is analysed, so say so when there are several.
    if (RESULT_WARNING_ELEMENT) {
        const facesFound = Number(result.faces_detected) || 1;
        let note = RESULT_WARNING_TEXT;
        if (verdict.report === "uncertain") {
            note += " This score is in the uncertain range: the model cannot say confidently either way.";
        }
        if (facesFound > 1) {
            note += " " + facesFound + " faces were found; only the largest face was analysed.";
        }
        RESULT_WARNING_ELEMENT.textContent = note;
    }

    if (result.face_box) {
        drawFaceBox(result.face_box);
    }
}


uploadAgainButton.addEventListener("click", function () {
    resetSelectedFile();
    mediaInput.click();
});


downloadReportButton.addEventListener("click", async function () {
    if (!selectedFile || !latestResult) {
        showError("Complete an image analysis before creating a report.");
        return;
    }

    const reportWindow = window.open("", "_blank");

    if (!reportWindow) {
        showError("The browser blocked the report window. Allow pop-ups and try again.");
        return;
    }

    reportWindow.document.write(
        "<p style='font-family:Arial;padding:30px'>Preparing report...</p>"
    );

    try {
        const imageData = await readFileAsDataURL(selectedFile);
        const checksum = await calculateChecksum(selectedFile);
        const imageDimensions = await getImageDimensions(imageData);
        const reportHTML = createReportHTML(imageData, checksum, imageDimensions);

        reportWindow.document.open();
        reportWindow.document.write(reportHTML);
        reportWindow.document.close();
    } catch (error) {
        reportWindow.close();
        showError("The report could not be created.");
    }
});


function readFileAsDataURL(file) {
    return new Promise(function (resolve, reject) {
        const reader = new FileReader();
        reader.onload = function () { resolve(reader.result); };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}


async function calculateChecksum(file) {
    const fileBuffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest("SHA-256", fileBuffer);
    return Array.from(new Uint8Array(hashBuffer))
        .map(function (byte) { return byte.toString(16).padStart(2, "0"); })
        .join("");
}


function getImageDimensions(imageData) {
    return new Promise(function (resolve, reject) {
        const image = new Image();
        image.onload = function () {
            resolve({ width: image.naturalWidth, height: image.naturalHeight });
        };
        image.onerror = reject;
        image.src = imageData;
    });
}


function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


const MODEL_INFO = {
    "detect_now_efficientnetb0_faceswap_crops_v2.keras": {
        description:
            "Fine-tuned on 2,748 face crops: about 1,400 Kaggle face-swap frames and 1,354 images " +
            "(677 pairs) made with the inswapper tool; 602 crops for validation. " +
            "Decision threshold 0.40, chosen by hand from validation results.",
        rows: `
        <div class="detail-row"><span>Kaggle test set (947 real, 947 swaps)</span><strong>88.0% of real images correct, 78.2% of swaps caught (83.1% overall)</strong></div>
        <div class="detail-row"><span>Inswapper test set (322 images)</span><strong>94% of real images correct, 72% of swaps caught (83% overall)</strong></div>
        <div class="detail-row"><span>Team test set (50 real, 50 swaps)</span><strong>92% of real images correct, 64% of swaps caught</strong></div>
        <p>The inswapper swaps come from the same tool used in training, so those tests do not show
        performance on tools the model has never seen. Expression edits made by AI tools are not detected
        (2 of 39 in the team test). Test sets are small, so treat small differences as noise.</p>`
    },
    "detect_now_efficientnetb0_faceswap_crops.keras": {
        description:
            "Fine-tuned on face swaps from one Kaggle dataset: 14,090 YuNet-prepared training faces " +
            "and 1,939 validation faces. Decision threshold 0.45.",
        rows: `
        <div class="detail-row"><span>Kaggle test set (947 real, 947 swaps)</span><strong>89.4% of real images correct, 92.9% of swaps caught (91.2% overall)</strong></div>
        <div class="detail-row"><span>Inswapper test set (322 images)</span><strong>78.3% of real images correct, 41.0% of swaps caught (59.6% overall)</strong></div>
        <div class="detail-row"><span>Team test set (50 real, 50 swaps)</span><strong>80% of real images correct, 34% of swaps caught</strong></div>
        <p>Trained on one swap tool only. It did not generalise to swaps made with the inswapper tool.
        Test sets are small, so treat small differences as noise.</p>`
    }
};


const REPORT_STYLE = `
* { box-sizing: border-box; }
body { margin: 0; background: #eef3f5; color: #233746; font-family: Arial, Helvetica, sans-serif; }
.report-actions { position: sticky; top: 0; z-index: 10; display: flex; padding: 14px 24px; justify-content: flex-end; gap: 10px; background: #0d2740; }
.report-actions button { padding: 11px 18px; background: #ffffff; color: #0d2740; border: none; border-radius: 7px; font-weight: bold; cursor: pointer; }
.report-actions .download { background: #1b829b; color: #ffffff; }
.report { width: min(1050px, calc(100% - 30px)); margin: 30px auto; }
.page { min-height: 1120px; margin-bottom: 24px; padding: 48px; background: #ffffff; border-radius: 4px; box-shadow: 0 12px 35px rgba(13, 39, 64, 0.10); break-after: page; }
.page:last-child { break-after: auto; }
.report-header { display: flex; padding-bottom: 24px; align-items: flex-start; justify-content: space-between; gap: 30px; border-bottom: 2px solid #e1eaee; }
.brand { display: flex; align-items: center; gap: 12px; }
.brand-icon { display: grid; width: 45px; height: 45px; place-items: center; background: linear-gradient(135deg, #126f8a, #5d6ee7); color: #ffffff; border-radius: 11px; font-weight: bold; }
.brand strong { display: block; color: #0d2740; font-size: 20px; }
.brand span { color: #718391; font-size: 11px; }
.header-details { text-align: right; }
.header-details strong { display: block; color: #0d2740; font-size: 13px; }
.header-details span { color: #718391; font-size: 11px; }
.report-title { margin: 32px 0 28px; }
.report-title span { color: #126f8a; font-size: 11px; font-weight: bold; letter-spacing: 2px; }
.report-title h1 { margin: 8px 0 6px; color: #0d2740; font-size: 30px; overflow-wrap: anywhere; }
.report-title p { margin: 0; color: #718391; font-size: 12px; }
.summary-grid { display: grid; margin-bottom: 28px; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.summary-card { min-height: 115px; padding: 17px; border: 1px solid #dce6eb; border-radius: 10px; }
.summary-card span { display: block; margin-bottom: 12px; color: #81919d; font-size: 9px; font-weight: bold; letter-spacing: 1px; }
.summary-card strong { display: block; color: #0d2740; font-size: 14px; overflow-wrap: anywhere; }
.summary-card small { display: block; margin-top: 7px; color: #718391; font-size: 10px; line-height: 1.4; }
.verdict { display: inline-block !important; padding: 6px 10px; border-radius: 20px; font-size: 11px !important; }
.verdict.real { background: #e2f5eb; color: #18764f; }
.verdict.fake { background: #ffe5e5; color: #a22d2d; }
.verdict.uncertain { background: #fff3d6; color: #8a5a00; }
.verdict.deepfake { background: #f7c4c4; color: #8c1d1d; }
.report-section { margin-top: 22px; overflow: hidden; border: 1px solid #dce6eb; border-radius: 11px; }
.report-section h2 { margin: 0; padding: 16px 18px; background: #f6f9fa; color: #0d2740; border-bottom: 1px solid #dce6eb; font-size: 15px; }
.detail-row { display: grid; padding: 12px 18px; grid-template-columns: 190px 1fr; border-bottom: 1px solid #edf1f3; }
.detail-row:last-child { border-bottom: none; }
.detail-row span { color: #81919d; font-size: 10px; font-weight: bold; letter-spacing: 0.6px; text-transform: uppercase; }
.detail-row strong, .detail-row p { margin: 0; color: #26394a; font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.probability-area { padding: 20px; }
.probability-item { margin-bottom: 18px; }
.probability-item:last-child { margin-bottom: 0; }
.probability-heading { display: flex; margin-bottom: 7px; justify-content: space-between; color: #26394a; font-size: 11px; font-weight: bold; }
.bar { height: 10px; overflow: hidden; background: #e8eef1; border-radius: 20px; }
.bar span { display: block; height: 100%; border-radius: 20px; }
.real-bar { background: #27966c; }
.fake-bar { background: #d95353; }
.media-preview { display: grid; min-height: 650px; padding: 30px; place-items: center; background: #f7f9fa; }
.media-preview img { display: block; max-width: 100%; max-height: 600px; object-fit: contain; border-radius: 6px; box-shadow: 0 8px 24px rgba(13, 39, 64, 0.12); }
.notice { margin-top: 25px; padding: 18px; background: #fff7dc; color: #685316; border: 1px solid #f0dfa2; border-radius: 9px; font-size: 11px; line-height: 1.6; }
.report-footer { display: flex; margin-top: 35px; padding-top: 15px; justify-content: space-between; color: #8b9aa5; border-top: 1px solid #e4ebee; font-size: 9px; }
@media print {
  @page { size: A4; margin: 0; }
  body { background: #ffffff; }
  .report-actions { display: none; }
  .report { width: 100%; margin: 0; }
  .page { width: 210mm; min-height: 297mm; margin: 0; padding: 14mm; border-radius: 0; box-shadow: none; }
}
@media (max-width: 750px) {
  .page { min-height: auto; padding: 25px; }
  .summary-grid { grid-template-columns: 1fr 1fr; }
  .detail-row { grid-template-columns: 1fr; gap: 6px; }
}
`;


function createReportHTML(imageData, checksum, imageDimensions) {
    const safeFileName = escapeHTML(selectedFile.name);

    const verdict = verdictInfo(latestResult);
    const safePrediction = verdict.label;

    const safeModel = escapeHTML(
        latestResult.model || latestResult.model_name || "Detect Now EfficientNetB0"
    );

    const modelFile = String(latestResult.model_file || "Unavailable in response");
    const info = MODEL_INFO[modelFile];
    const fakeScore = Number(latestResult.fake_probability);
    const realScore = Number(latestResult.real_probability);
    const fakeThreshold = Number(latestResult.fake_threshold);
    const bandA = Number(latestResult.band_real_below);
    const bandB = Number(latestResult.band_uncertain_below);
    const bandC = Number(latestResult.band_deepfake_from);
    const bandsKnown = Number.isFinite(bandA) && Number.isFinite(bandB) && Number.isFinite(bandC) && Boolean(latestResult.verdict);
    const expectedVerdict =
        fakeScore < bandA ? "LIKELY_REAL" : fakeScore < bandB ? "UNCERTAIN" : fakeScore < bandC ? "LIKELY_DEEPFAKE" : "DEEPFAKE";
    const mismatch = bandsKnown
        ? Number.isFinite(fakeScore) && expectedVerdict !== latestResult.verdict
        : Number.isFinite(fakeScore) &&
          Number.isFinite(fakeThreshold) &&
          (safePrediction === "DEEPFAKE" ? fakeScore < fakeThreshold : fakeScore >= fakeThreshold);

    const safeBarWidth = function (value) {
        return Math.min(100, Math.max(0, Number.isFinite(value) ? value : 0));
    };

    const datasetDescription = info
        ? info.description
        : "Training dataset details are unavailable for this backend model.";

    const evaluationRows = info
        ? info.rows
        : "<p>No verified evaluation figures are available for this model.</p>";

    const analysisDate = new Date(latestResult.analysedAt).toLocaleString();
    const fileSize = formatFileSize(selectedFile.size);
    const verdictClass = verdict.report;
    const facesFound = Number(latestResult.faces_detected) || 1;

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detect Now Report - ${safeFileName}</title>
<style>${REPORT_STYLE}</style>
</head>
<body>
<div class="report-actions">
  <button onclick="window.close()">Close</button>
  <button class="download" onclick="window.print()">Print or Save as PDF</button>
</div>
<main class="report">

<section class="page">
  <header class="report-header">
    <div class="brand">
      <span class="brand-icon">D</span>
      <div><strong>Detect Now</strong><span>Deepfake Detection Prototype</span></div>
    </div>
    <div class="header-details">
      <strong>DETECTION REPORT</strong>
      <span>${escapeHTML(analysisDate)}</span>
    </div>
  </header>

  <div class="report-title">
    <span>IMAGE ANALYSIS REPORT</span>
    <h1>${safeFileName}</h1>
    <p>Image · Analysed ${escapeHTML(analysisDate)}</p>
  </div>

  <div class="summary-grid">
    <div class="summary-card"><span>VERDICT</span><strong class="verdict ${verdictClass}">${safePrediction}</strong></div>
    <div class="summary-card"><span>MEDIA</span><strong>Image</strong><small>${safeFileName}<br>${escapeHTML(fileSize)}</small></div>
    <div class="summary-card"><span>SUBMITTED</span><strong>${escapeHTML(analysisDate)}</strong><small>Human face detected</small></div>
    <div class="summary-card"><span>DEEPFAKE SCORE</span><strong>${fakeScore.toFixed(2)}%</strong><small>Raw model score</small></div>
  </div>

  <section class="report-section">
    <h2>File Details</h2>
    <div class="detail-row"><span>File type</span><strong>${escapeHTML(selectedFile.type)}</strong></div>
    <div class="detail-row"><span>File name</span><strong>${safeFileName}</strong></div>
    <div class="detail-row"><span>File size</span><strong>${escapeHTML(fileSize)}</strong></div>
    <div class="detail-row"><span>Dimensions</span><strong>${imageDimensions.width} × ${imageDimensions.height} pixels</strong></div>
    <div class="detail-row"><span>Analysis date</span><strong>${escapeHTML(analysisDate)}</strong></div>
    <div class="detail-row"><span>Face detected</span><strong>Yes${facesFound > 1 ? " (" + facesFound + " faces found; largest analysed)" : ""}</strong></div>
    <div class="detail-row"><span>SHA-256 checksum</span><strong>${escapeHTML(checksum)}</strong></div>
  </section>

  <section class="report-section">
    <h2>Deepfake Detection Results</h2>
    <div class="detail-row"><span>Verdict</span><strong>${safePrediction}</strong></div>
    <div class="detail-row"><span>Deepfake score</span><strong>${fakeScore.toFixed(2)}%</strong></div>
    <div class="detail-row"><span>Model file</span><strong>${escapeHTML(modelFile)}</strong></div>
    <div class="detail-row"><span>Verdict bands</span><strong>${escapeHTML(bandsKnown ? bandText(latestResult) : (Number.isFinite(fakeThreshold) ? "Decision threshold " + fakeThreshold.toFixed(2) + "%" : "Unavailable"))}</strong></div>
    <div class="detail-row"><span>Model</span><strong>${safeModel}</strong></div>
    <div class="detail-row"><span>Architecture</span><strong>EfficientNetB0</strong></div>
    <div class="detail-row"><span>Face validation</span><strong>Passed - human face detected</strong></div>

    <div class="probability-area">
      <div class="probability-item">
        <div class="probability-heading"><span>Real score</span><span>${realScore.toFixed(2)}%</span></div>
        <div class="bar"><span class="real-bar" style="width: ${safeBarWidth(realScore)}%"></span></div>
      </div>
      <div class="probability-item">
        <div class="probability-heading"><span>Deepfake score</span><span>${fakeScore.toFixed(2)}%</span></div>
        <div class="bar"><span class="fake-bar" style="width: ${safeBarWidth(fakeScore)}%"></span></div>
      </div>
    </div>
  </section>

  <div class="notice">
    <strong>Important:</strong>
    ${mismatch ? "Warning: the verdict does not match the score and bands. Restart the backend and analyse again." : ""}
    This report contains an experimental prediction from our trained EfficientNetB0
    model. It should not be treated as forensic proof or used as the only evidence
    for an important decision.
  </div>

  <div class="report-footer"><span>Detect Now · Group 20</span><span>Page 1 of 2</span></div>
</section>

<section class="page">
  <header class="report-header">
    <div class="brand">
      <span class="brand-icon">D</span>
      <div><strong>Detect Now</strong><span>Deepfake Detection Prototype</span></div>
    </div>
    <div class="header-details"><strong>MEDIA REVIEW</strong><span>${safeFileName}</span></div>
  </header>

  <section class="report-section">
    <h2>Media Preview</h2>
    <div class="media-preview"><img src="${imageData}" alt="Analysed image"></div>
  </section>

  <section class="report-section">
    <h2>Analysis Information</h2>
    <div class="detail-row"><span>Analysis method</span>
      <p>YuNet locates the face. The backend crops around it, pads it to a square
      and resizes it to 224 × 224 pixels for EfficientNetB0.</p></div>
    <div class="detail-row"><span>Training images</span><p>${escapeHTML(datasetDescription)}</p></div>
    <div class="detail-row"><span>Training method</span><p>Transfer learning and EfficientNetB0 fine-tuning.</p></div>
    <div class="detail-row"><span>Score interpretation</span>
      <p>The deepfake score is a raw classifier output, not a calibrated probability.
      The verdict comes from fixed score bands (see Verdict bands above); the uncertain band means the model is unsure.</p></div>
    ${evaluationRows}
  </section>

  <section class="report-section">
    <h2>Known Limitations</h2>
    <div class="detail-row"><span>Image quality</span><p>Compression, blur, lighting and low resolution may affect the prediction.</p></div>
    <div class="detail-row"><span>Face position</span><p>Side-facing, covered or very small faces may not be detected correctly.</p></div>
    <div class="detail-row"><span>Model coverage</span><p>Deepfake methods not represented in the training dataset may be more difficult to identify.</p></div>
    <div class="detail-row"><span>Interpretation</span><p>Model confidence is not the same as guaranteed accuracy.</p></div>
  </section>

  <div class="report-footer"><span>Charles Darwin University Academic Prototype</span><span>Page 2 of 2</span></div>
</section>

</main>
</body>
</html>`;
}


function formatFileSize(fileSize) {
    if (fileSize < 1024 * 1024) {
        return (fileSize / 1024).toFixed(1) + " KB";
    }
    return (fileSize / (1024 * 1024)).toFixed(1) + " MB";
}


// ----- Upload history (stored only in this browser) -----------------------

function readStorage(key) {
    try {
        return localStorage.getItem(key);
    } catch (error) {
        return null; // storage blocked (private mode, browser setting)
    }
}


function writeStorage(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (error) {
        // ignore: history is optional
    }
}


function getUploadHistory() {
    const savedHistory = readStorage(HISTORY_KEY);

    if (!savedHistory) {
        return [];
    }

    try {
        const parsed = JSON.parse(savedHistory);
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        try {
            localStorage.removeItem(HISTORY_KEY);
        } catch (removeError) {
            // ignore
        }
        return [];
    }
}


function saveHistoryItem(fileName, mediaType, fileSize) {
    const uploadHistory = getUploadHistory();

    uploadHistory.unshift({
        name: fileName,
        type: mediaType,
        size: formatFileSize(fileSize),
        date: new Date().toLocaleString()
    });

    writeStorage(HISTORY_KEY, JSON.stringify(uploadHistory.slice(0, 10)));
    displayUploadHistory();
}


function displayUploadHistory() {
    const uploadHistory = getUploadHistory();

    historyList.innerHTML = "";

    if (uploadHistory.length === 0) {
        const emptyContainer = document.createElement("div");
        const emptyIcon = document.createElement("span");
        const emptyHeading = document.createElement("h3");
        const emptyText = document.createElement("p");

        emptyContainer.className = "empty-history-message";
        emptyIcon.textContent = "◷";
        emptyHeading.textContent = "No upload history";
        emptyText.textContent = "Your recently selected files will appear here.";

        emptyContainer.appendChild(emptyIcon);
        emptyContainer.appendChild(emptyHeading);
        emptyContainer.appendChild(emptyText);
        historyList.appendChild(emptyContainer);
        return;
    }

    uploadHistory.forEach(function (item, index) {
        const historyItem = document.createElement("div");
        const fileContainer = document.createElement("div");
        const fileName = document.createElement("strong");
        const fileDetails = document.createElement("small");
        const actionContainer = document.createElement("div");
        const mediaTypeLabel = document.createElement("span");
        const deleteButton = document.createElement("button");

        actionContainer.style.display = "flex";
        actionContainer.style.alignItems = "center";
        actionContainer.style.gap = "8px";

        deleteButton.type = "button";
        deleteButton.innerHTML = "&times;";
        deleteButton.title = "Delete item";
        deleteButton.style.background = "transparent";
        deleteButton.style.border = "none";
        deleteButton.style.color = "#a22d2d";
        deleteButton.style.fontSize = "16px";
        deleteButton.style.fontWeight = "bold";
        deleteButton.style.cursor = "pointer";
        deleteButton.style.padding = "2px 6px";
        deleteButton.style.borderRadius = "4px";

        deleteButton.addEventListener("click", function (event) {
            event.stopPropagation();
            deleteHistoryItem(index);
        });

        historyItem.className = "history-item";
        historyItem.style.cursor = "pointer";

        historyItem.addEventListener("click", function () {
            openUploadArea(item.type === "video" ? "video" : "image");
            document.getElementById("detector").scrollIntoView({ behavior: "smooth" });
        });

        fileName.textContent = item.name;
        fileDetails.textContent = item.size + " · " + item.date;

        mediaTypeLabel.className = "history-type";
        mediaTypeLabel.textContent = item.type === "image" ? "Image" : "Video";

        fileContainer.appendChild(fileName);
        fileContainer.appendChild(fileDetails);
        historyItem.appendChild(fileContainer);

        actionContainer.appendChild(mediaTypeLabel);
        actionContainer.appendChild(deleteButton);
        historyItem.appendChild(actionContainer);

        historyList.appendChild(historyItem);
    });
}


function deleteHistoryItem(index) {
    const uploadHistory = getUploadHistory();
    uploadHistory.splice(index, 1);
    writeStorage(HISTORY_KEY, JSON.stringify(uploadHistory));
    displayUploadHistory();
}


clearHistoryButton.addEventListener("click", function () {
    try {
        localStorage.removeItem(HISTORY_KEY);
    } catch (error) {
        // ignore
    }
    displayUploadHistory();
});


// ----- Drag and drop -------------------------------------------------------

const uploadBox = document.querySelector(".upload-box");

if (uploadBox) {
    ["dragenter", "dragover"].forEach(function (eventName) {
        uploadBox.addEventListener(eventName, function (event) {
            event.preventDefault();
            event.stopPropagation();
            uploadBox.classList.add("drag-over");
        }, false);
    });

    ["dragleave", "drop"].forEach(function (eventName) {
        uploadBox.addEventListener(eventName, function (event) {
            event.preventDefault();
            event.stopPropagation();
            uploadBox.classList.remove("drag-over");
        }, false);
    });

    uploadBox.addEventListener("drop", function (event) {
        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            mediaInput.files = files;
            mediaInput.dispatchEvent(new Event("change"));
        }
    });
}


// ----- Face box overlay ----------------------------------------------------
// The backend returns the face position in the ORIGINAL image's pixels. The
// canvas is created here and laid over the preview image.

function drawFaceBox(faceBox) {
    if (!faceBox || !currentPreviewURL) {
        return;
    }

    const sourceUrl = currentPreviewURL;
    const source = new Image();

    source.onload = function () {
        if (sourceUrl !== currentPreviewURL) {
            return; // another file was chosen in the meantime
        }

        const longest = Math.max(source.naturalWidth, source.naturalHeight);
        const scale = Math.min(1, 1024 / longest);
        const width = Math.round(source.naturalWidth * scale);
        const height = Math.round(source.naturalHeight * scale);

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        context.drawImage(source, 0, 0, width, height);

        // face_box is in the photo's own pixels, so it lines up exactly on the copy
        const x = faceBox.x * scale;
        const y = faceBox.y * scale;
        const boxWidth = faceBox.width * scale;
        const boxHeight = faceBox.height * scale;

        context.strokeStyle = "#35a6aa";
        context.lineWidth = Math.max(2, Math.round(Math.max(width, height) / 200));
        context.strokeRect(x, y, boxWidth, boxHeight);

        const fontSize = Math.max(11, Math.round(Math.max(width, height) / 40));
        const label = "Face detected";
        context.font = "bold " + fontSize + "px sans-serif";
        const labelWidth = context.measureText(label).width + fontSize;
        const labelHeight = Math.round(fontSize * 1.5);
        const labelY = y - labelHeight >= 0 ? y - labelHeight : y;
        context.fillStyle = "#126f8a";
        context.fillRect(x, labelY, labelWidth, labelHeight);
        context.fillStyle = "#ffffff";
        context.fillText(label, x + fontSize / 2, labelY + labelHeight * 0.72);

        imagePreview.src = canvas.toDataURL("image/png");
        imagePreview.dataset.boxed = "1";
    };

    source.src = sourceUrl;
}


// ----- History export ------------------------------------------------------

function csvCell(value) {
    let text = String(value ?? "");
    // stop spreadsheet programs treating a file name as a formula
    if (/^[=+\-@\t\r]/.test(text)) {
        text = "'" + text;
    }
    return '"' + text.replaceAll('"', '""') + '"';
}


const exportCsvButton = document.getElementById("export-csv-button");

if (exportCsvButton) {
    exportCsvButton.addEventListener("click", function () {
        const history = getUploadHistory();

        if (history.length === 0) {
            alert("No history available to export.");
            return;
        }

        const rows = ["File Name,File Size,Type,Date"];
        history.forEach(function (item) {
            rows.push(
                [item.name, item.size, item.type, item.date].map(csvCell).join(",")
            );
        });

        const blob = new Blob(["\ufeff" + rows.join("\n")], {
            type: "text/csv;charset=utf-8"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "detect_now_upload_history_" + Date.now() + ".csv";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });
}


displayUploadHistory();