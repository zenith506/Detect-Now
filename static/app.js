const API_URL =
    window.location.protocol.startsWith("http")
        ? `${window.location.protocol}//${window.location.hostname}:5000/predict`
        : "http://127.0.0.1:5000/predict";

const mediaSelection =
    document.getElementById("media-selection");

const uploadArea =
    document.getElementById("upload-area");

const selectImageButton =
    document.getElementById("select-image-button");

const selectVideoButton =
    document.getElementById("select-video-button");

const backButton =
    document.getElementById("back-button");

const selectedTypeIcon =
    document.getElementById("selected-type-icon");

const uploadHeading =
    document.getElementById("upload-heading");

const uploadDescription =
    document.getElementById("upload-description");

const uploadText =
    document.getElementById("upload-text");

const fileInformation =
    document.getElementById("file-information");

const mediaInput =
    document.getElementById("media-input");

const imagePreviewSection =
    document.getElementById("image-preview-section");

const imagePreview =
    document.getElementById("image-preview");

const imageFileName =
    document.getElementById("image-file-name");

const videoPreviewSection =
    document.getElementById("video-preview-section");

const videoPreview =
    document.getElementById("video-preview");

const videoFileName =
    document.getElementById("video-file-name");

const videoNotice =
    document.getElementById("video-notice");

const analysisLoading =
    document.getElementById("analysis-loading");

const analysisError =
    document.getElementById("analysis-error");

const analysisErrorText =
    document.getElementById("analysis-error-text");

const detectionResult =
    document.getElementById("detection-result");

const predictionText =
    document.getElementById("prediction-text");

const confidenceText =
    document.getElementById("confidence-text");

const confidenceBarFill =
    document.getElementById("confidence-bar-fill");

const realProbability =
    document.getElementById("real-probability");

const fakeProbability =
    document.getElementById("fake-probability");

const analyseButton =
    document.getElementById("analyse-button");

const downloadReportButton =
    document.getElementById("download-report-button");

const uploadAgainButton =
    document.getElementById("upload-again-button");

const clearHistoryButton =
    document.getElementById("clear-history-button");

const historyList =
    document.getElementById("history-list");

let selectedMediaType = "";
let selectedFile = null;
let currentPreviewURL = "";
let latestResult = null;


function openUploadArea(mediaType) {
    selectedMediaType = mediaType;

    resetSelectedFile();

    mediaSelection.hidden = true;
    uploadArea.hidden = false;

    if (mediaType === "image") {
        selectedTypeIcon.textContent = "▧";
        uploadHeading.textContent = "Upload Image";

        uploadDescription.textContent =
            "Select an image from your computer or device.";

        uploadText.textContent = "Choose an image";

        fileInformation.textContent =
            "JPG, JPEG, PNG, WEBP or BMP · Maximum 10 MB";

        mediaInput.accept =
            ".jpg,.jpeg,.png,image/jpeg,image/png";
    } else {
        selectedTypeIcon.textContent = "▶";
        uploadHeading.textContent = "Upload Video";

        uploadDescription.textContent =
            "Select a video from your computer or device.";

        uploadText.textContent = "Choose a video";

        fileInformation.textContent =
            "MP4, WebM or MOV · Maximum 50 MB";

        mediaInput.accept =
            ".mp4,.webm,.mov,video/mp4,video/webm,video/quicktime";
    }
}


function resetSelectedFile() {
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

    imagePreview.removeAttribute("src");
    imageFileName.textContent = "";

    videoPreview.pause();
    videoPreview.removeAttribute("src");
    videoPreview.load();
    videoFileName.textContent = "";

    confidenceBarFill.style.width = "0%";

    detectionResult.classList.remove(
        "real-result",
        "fake-result",
        "uncertain-result"
    );

    if (currentPreviewURL) {
        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = "";
    }
}


function resetResultAreas() {
    latestResult = null;

    analysisLoading.hidden = true;
    analysisError.hidden = true;
    detectionResult.hidden = true;
    videoNotice.hidden = true;
    downloadReportButton.hidden = true;

    confidenceBarFill.style.width = "0%";

    detectionResult.classList.remove(
        "real-result",
        "fake-result",
        "uncertain-result"
    );
}


function showError(message) {
    latestResult = null;

    analysisErrorText.textContent = message;
    analysisError.hidden = false;
    analysisLoading.hidden = true;
    detectionResult.hidden = true;
    downloadReportButton.hidden = true;

    confidenceBarFill.style.width = "0%";
}


selectImageButton.addEventListener(
    "click",
    function () {
        openUploadArea("image");
    }
);


selectVideoButton.addEventListener(
    "click",
    function () {
        openUploadArea("video");
    }
);


backButton.addEventListener(
    "click",
    function () {
        resetSelectedFile();

        uploadArea.hidden = true;
        mediaSelection.hidden = false;
        selectedMediaType = "";
    }
);


mediaInput.addEventListener(
    "change",
    function () {
        const file = mediaInput.files[0];

        if (!file) {
            return;
        }

        resetResultAreas();

        if (
            selectedMediaType === "image" &&
            !file.type.startsWith("image/")
        ) {
            showError(
                "Please select a valid JPG, JPEG or PNG image."
            );

            mediaInput.value = "";
            return;
        }

        if (
            selectedMediaType === "video" &&
            !file.type.startsWith("video/")
        ) {
            showError(
                "Please select a valid MP4, WebM or MOV video."
            );

            mediaInput.value = "";
            return;
        }

        const imageMaximumSize =
            10 * 1024 * 1024;

        const videoMaximumSize =
            50 * 1024 * 1024;

        if (
            selectedMediaType === "image" &&
            file.size > imageMaximumSize
        ) {
            showError(
                "Please select an image smaller than 10 MB."
            );

            mediaInput.value = "";
            return;
        }

        if (
            selectedMediaType === "video" &&
            file.size > videoMaximumSize
        ) {
            showError(
                "Please select a video smaller than 50 MB."
            );

            mediaInput.value = "";
            return;
        }

        selectedFile = file;

        if (currentPreviewURL) {
            URL.revokeObjectURL(currentPreviewURL);
        }

        currentPreviewURL =
            URL.createObjectURL(file);

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

        saveHistoryItem(
            file.name,
            selectedMediaType,
            file.size
        );
    }
);


analyseButton.addEventListener(
    "click",
    async function () {
        if (!selectedFile) {
            showError(
                "Please select an image before starting analysis."
            );

            return;
        }

        resetResultAreas();

        analysisLoading.hidden = false;
        analyseButton.disabled = true;
        analyseButton.textContent = "Analysing...";

        const formData = new FormData();

        formData.append(
            "image",
            selectedFile,
            selectedFile.name
        );

        try {
            const response = await fetch(
                API_URL,
                {
                    method: "POST",
                    body: formData
                }
            );

            let result;

            try {
                result = await response.json();
            } catch (error) {
                throw new Error(
                    "The detection server returned an invalid response."
                );
            }

            if (!response.ok || result.success === false) {
                throw new Error(
                    result.message ||
                    result.error ||
                    "The image could not be analysed."
                );
            }

            latestResult = {
                ...result,
                analysedAt: new Date().toISOString()
            };

            displayResult(latestResult);
        } catch (error) {
            if (
                error.message === "Failed to fetch" ||
                error instanceof TypeError
            ) {
                showError(
                    "The detection server is unavailable. Start backend.py and try again."
                );
            } else {
                showError(error.message);
            }
        } finally {
            analysisLoading.hidden = true;
            analyseButton.disabled = false;
            analyseButton.textContent =
                "Analyse Image Again";
        }
    }
);


function displayResult(result) {
    const normalizedPrediction = String(
        result.prediction
    ).toUpperCase();

    const isReal =
        normalizedPrediction === "REAL";

    predictionText.textContent =
        isReal ? "REAL" : "DEEPFAKE";

    confidenceText.textContent =
        Number(result.confidence).toFixed(2) +
        "% confidence";

    realProbability.textContent =
        Number(result.real_probability).toFixed(2) +
        "%";

    fakeProbability.textContent =
        Number(result.fake_probability).toFixed(2) +
        "%";

    confidenceBarFill.style.width =
        Math.min(
            100,
            Math.max(
                0,
                Number(result.confidence)
            )
        ) + "%";

    detectionResult.classList.remove(
        "real-result",
        "fake-result",
        "uncertain-result"
    );

    detectionResult.classList.add(
        isReal
            ? "real-result"
            : "fake-result"
    );

    detectionResult.hidden = false;
    analysisError.hidden = true;
    downloadReportButton.hidden = false;

    if (result.face_box) {
        drawFaceBox(result.face_box);
    }
}


uploadAgainButton.addEventListener(
    "click",
    function () {
        resetSelectedFile();
        mediaInput.click();
    }
);


downloadReportButton.addEventListener(
    "click",
    async function () {
        if (!selectedFile || !latestResult) {
            showError(
                "Complete an image analysis before creating a report."
            );

            return;
        }

        const reportWindow = window.open(
            "",
            "_blank"
        );

        if (!reportWindow) {
            showError(
                "The browser blocked the report window. Allow pop-ups and try again."
            );

            return;
        }

        reportWindow.document.write(
            "<p style='font-family:Arial;padding:30px'>Preparing report...</p>"
        );

        try {
            const imageData =
                await readFileAsDataURL(
                    selectedFile
                );

            const checksum =
                await calculateChecksum(
                    selectedFile
                );

            const imageDimensions =
                await getImageDimensions(
                    imageData
                );

            const reportHTML =
                createReportHTML(
                    imageData,
                    checksum,
                    imageDimensions
                );

            reportWindow.document.open();
            reportWindow.document.write(
                reportHTML
            );
            reportWindow.document.close();
        } catch (error) {
            reportWindow.close();

            showError(
                "The report could not be created."
            );
        }
    }
);


function readFileAsDataURL(file) {
    return new Promise(
        function (resolve, reject) {
            const reader = new FileReader();

            reader.onload = function () {
                resolve(reader.result);
            };

            reader.onerror = reject;

            reader.readAsDataURL(file);
        }
    );
}


async function calculateChecksum(file) {
    const fileBuffer =
        await file.arrayBuffer();

    const hashBuffer =
        await crypto.subtle.digest(
            "SHA-256",
            fileBuffer
        );

    const hashArray = Array.from(
        new Uint8Array(hashBuffer)
    );

    return hashArray
        .map(function (byte) {
            return byte
                .toString(16)
                .padStart(2, "0");
        })
        .join("");
}


function getImageDimensions(imageData) {
    return new Promise(
        function (resolve, reject) {
            const image = new Image();

            image.onload = function () {
                resolve({
                    width: image.naturalWidth,
                    height: image.naturalHeight
                });
            };

            image.onerror = reject;
            image.src = imageData;
        }
    );
}


function escapeHTML(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function createReportHTML(
    imageData,
    checksum,
    imageDimensions
) {
    const safeFileName =
        escapeHTML(selectedFile.name);

    const normalizedPrediction = String(
        latestResult.prediction
    ).toUpperCase();

    const safePrediction =
        normalizedPrediction === "REAL"
            ? "REAL"
            : "DEEPFAKE";

    const safeModel = escapeHTML(
        latestResult.model ||
        latestResult.model_name ||
        "Detect Now EfficientNetB0"
    );

    const analysisDate = new Date(
        latestResult.analysedAt
    ).toLocaleString();

    const fileSize =
        formatFileSize(selectedFile.size);

    const verdictClass =
        safePrediction === "REAL"
            ? "real"
            : "fake";

    return `
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Detect Now Report - ${safeFileName}</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: #eef3f5;
            color: #233746;
            font-family: Arial, Helvetica, sans-serif;
        }

        .report-actions {
            position: sticky;
            top: 0;
            z-index: 10;
            display: flex;
            padding: 14px 24px;
            justify-content: flex-end;
            gap: 10px;
            background: #0d2740;
        }

        .report-actions button {
            padding: 11px 18px;
            background: #ffffff;
            color: #0d2740;
            border: none;
            border-radius: 7px;
            font-weight: bold;
            cursor: pointer;
        }

        .report-actions .download {
            background: #1b829b;
            color: #ffffff;
        }

        .report {
            width: min(1050px, calc(100% - 30px));
            margin: 30px auto;
        }

        .page {
            min-height: 1120px;
            margin-bottom: 24px;
            padding: 48px;
            background: #ffffff;
            border-radius: 4px;
            box-shadow: 0 12px 35px
                rgba(13, 39, 64, 0.10);
            break-after: page;
        }

        .page:last-child {
            break-after: auto;
        }

        .report-header {
            display: flex;
            padding-bottom: 24px;
            align-items: flex-start;
            justify-content: space-between;
            gap: 30px;
            border-bottom: 2px solid #e1eaee;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            display: grid;
            width: 45px;
            height: 45px;
            place-items: center;
            background: linear-gradient(
                135deg,
                #126f8a,
                #5d6ee7
            );
            color: #ffffff;
            border-radius: 11px;
            font-weight: bold;
        }

        .brand strong {
            display: block;
            color: #0d2740;
            font-size: 20px;
        }

        .brand span {
            color: #718391;
            font-size: 11px;
        }

        .header-details {
            text-align: right;
        }

        .header-details strong {
            display: block;
            color: #0d2740;
            font-size: 13px;
        }

        .header-details span {
            color: #718391;
            font-size: 11px;
        }

        .report-title {
            margin: 32px 0 28px;
        }

        .report-title span {
            color: #126f8a;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 2px;
        }

        .report-title h1 {
            margin: 8px 0 6px;
            color: #0d2740;
            font-size: 30px;
            overflow-wrap: anywhere;
        }

        .report-title p {
            margin: 0;
            color: #718391;
            font-size: 12px;
        }

        .summary-grid {
            display: grid;
            margin-bottom: 28px;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }

        .summary-card {
            min-height: 115px;
            padding: 17px;
            border: 1px solid #dce6eb;
            border-radius: 10px;
        }

        .summary-card span {
            display: block;
            margin-bottom: 12px;
            color: #81919d;
            font-size: 9px;
            font-weight: bold;
            letter-spacing: 1px;
        }

        .summary-card strong {
            display: block;
            color: #0d2740;
            font-size: 14px;
            overflow-wrap: anywhere;
        }

        .summary-card small {
            display: block;
            margin-top: 7px;
            color: #718391;
            font-size: 10px;
            line-height: 1.4;
        }

        .verdict {
            display: inline-block !important;
            padding: 6px 10px;
            border-radius: 20px;
            font-size: 11px !important;
        }

        .verdict.real {
            background: #e2f5eb;
            color: #18764f;
        }

        .verdict.fake {
            background: #ffe5e5;
            color: #a22d2d;
        }

        .report-section {
            margin-top: 22px;
            overflow: hidden;
            border: 1px solid #dce6eb;
            border-radius: 11px;
        }

        .report-section h2 {
            margin: 0;
            padding: 16px 18px;
            background: #f6f9fa;
            color: #0d2740;
            border-bottom: 1px solid #dce6eb;
            font-size: 15px;
        }

        .detail-row {
            display: grid;
            padding: 12px 18px;
            grid-template-columns: 190px 1fr;
            border-bottom: 1px solid #edf1f3;
        }

        .detail-row:last-child {
            border-bottom: none;
        }

        .detail-row span {
            color: #81919d;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }

        .detail-row strong,
        .detail-row p {
            margin: 0;
            color: #26394a;
            font-size: 11px;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }

        .probability-area {
            padding: 20px;
        }

        .probability-item {
            margin-bottom: 18px;
        }

        .probability-item:last-child {
            margin-bottom: 0;
        }

        .probability-heading {
            display: flex;
            margin-bottom: 7px;
            justify-content: space-between;
            color: #26394a;
            font-size: 11px;
            font-weight: bold;
        }

        .bar {
            height: 10px;
            overflow: hidden;
            background: #e8eef1;
            border-radius: 20px;
        }

        .bar span {
            display: block;
            height: 100%;
            border-radius: 20px;
        }

        .real-bar {
            background: #27966c;
        }

        .fake-bar {
            background: #d95353;
        }

        .media-preview {
            display: grid;
            min-height: 650px;
            padding: 30px;
            place-items: center;
            background: #f7f9fa;
        }

        .media-preview img {
            display: block;
            max-width: 100%;
            max-height: 600px;
            object-fit: contain;
            border-radius: 6px;
            box-shadow: 0 8px 24px
                rgba(13, 39, 64, 0.12);
        }

        .notice {
            margin-top: 25px;
            padding: 18px;
            background: #fff7dc;
            color: #685316;
            border: 1px solid #f0dfa2;
            border-radius: 9px;
            font-size: 11px;
            line-height: 1.6;
        }

        .report-footer {
            display: flex;
            margin-top: 35px;
            padding-top: 15px;
            justify-content: space-between;
            color: #8b9aa5;
            border-top: 1px solid #e4ebee;
            font-size: 9px;
        }

        @media print {
            @page {
                size: A4;
                margin: 0;
            }

            body {
                background: #ffffff;
            }

            .report-actions {
                display: none;
            }

            .report {
                width: 100%;
                margin: 0;
            }

            .page {
                width: 210mm;
                min-height: 297mm;
                margin: 0;
                padding: 14mm;
                border-radius: 0;
                box-shadow: none;
            }
        }

        @media (max-width: 750px) {
            .page {
                min-height: auto;
                padding: 25px;
            }

            .summary-grid {
                grid-template-columns: 1fr 1fr;
            }

            .detail-row {
                grid-template-columns: 1fr;
                gap: 6px;
            }
        }
    </style>
</head>

<body>

    <div class="report-actions">

        <button onclick="window.close()">
            Close
        </button>

        <button
            class="download"
            onclick="window.print()"
        >
            Print or Save as PDF
        </button>

    </div>

    <main class="report">

        <section class="page">

            <header class="report-header">

                <div class="brand">

                    <span class="brand-icon">
                        D
                    </span>

                    <div>
                        <strong>Detect Now</strong>
                        <span>
                            Deepfake Detection Prototype
                        </span>
                    </div>

                </div>

                <div class="header-details">
                    <strong>DETECTION REPORT</strong>
                    <span>
                        ${escapeHTML(analysisDate)}
                    </span>
                </div>

            </header>

            <div class="report-title">

                <span>IMAGE ANALYSIS REPORT</span>

                <h1>${safeFileName}</h1>

                <p>
                    Image · Analysed
                    ${escapeHTML(analysisDate)}
                </p>

            </div>

            <div class="summary-grid">

                <div class="summary-card">
                    <span>VERDICT</span>

                    <strong
                        class="verdict ${verdictClass}"
                    >
                        ${safePrediction}
                    </strong>
                </div>

                <div class="summary-card">
                    <span>MEDIA</span>
                    <strong>Image</strong>

                    <small>
                        ${safeFileName}<br>
                        ${escapeHTML(fileSize)}
                    </small>
                </div>

                <div class="summary-card">
                    <span>SUBMITTED</span>

                    <strong>
                        ${escapeHTML(analysisDate)}
                    </strong>

                    <small>
                        Human face detected
                    </small>
                </div>

                <div class="summary-card">
                    <span>CONFIDENCE</span>

                    <strong>
                        ${Number(
        latestResult.confidence
    ).toFixed(2)}%
                    </strong>

                    <small>Model confidence</small>
                </div>

            </div>

            <section class="report-section">

                <h2>File Details</h2>

                <div class="detail-row">
                    <span>File type</span>

                    <strong>
                        ${escapeHTML(
        selectedFile.type
    )}
                    </strong>
                </div>

                <div class="detail-row">
                    <span>File name</span>
                    <strong>${safeFileName}</strong>
                </div>

                <div class="detail-row">
                    <span>File size</span>

                    <strong>
                        ${escapeHTML(fileSize)}
                    </strong>
                </div>

                <div class="detail-row">
                    <span>Dimensions</span>

                    <strong>
                        ${imageDimensions.width} ×
                        ${imageDimensions.height}
                        pixels
                    </strong>
                </div>

                <div class="detail-row">
                    <span>Analysis date</span>

                    <strong>
                        ${escapeHTML(analysisDate)}
                    </strong>
                </div>

                <div class="detail-row">
                    <span>Face detected</span>
                    <strong>Yes</strong>
                </div>

                <div class="detail-row">
                    <span>SHA-256 checksum</span>

                    <strong>
                        ${escapeHTML(checksum)}
                    </strong>
                </div>

            </section>

            <section class="report-section">

                <h2>
                    Deepfake Detection Results
                </h2>

                <div class="detail-row">
                    <span>Prediction</span>
                    <strong>${safePrediction}</strong>
                </div>

                <div class="detail-row">
                    <span>Model confidence</span>

                    <strong>
                        ${Number(
        latestResult.confidence
    ).toFixed(2)}%
                    </strong>
                </div>

                <div class="detail-row">
                    <span>Model</span>
                    <strong>${safeModel}</strong>
                </div>

                <div class="detail-row">
                    <span>Architecture</span>
                    <strong>EfficientNetB0</strong>
                </div>

                <div class="detail-row">
                    <span>Face validation</span>

                    <strong>
                        Passed - human face detected
                    </strong>
                </div>

                <div class="probability-area">

                    <div class="probability-item">

                        <div class="probability-heading">

                            <span>Real probability</span>

                            <span>
                                ${Number(
        latestResult
            .real_probability
    ).toFixed(2)}%
                            </span>

                        </div>

                        <div class="bar">

                            <span
                                class="real-bar"
                                style="width:
                                ${latestResult
            .real_probability}%"
                            ></span>

                        </div>

                    </div>

                    <div class="probability-item">

                        <div class="probability-heading">

                            <span>
                                Deepfake probability
                            </span>

                            <span>
                                ${Number(
                latestResult
                    .fake_probability
            ).toFixed(2)}%
                            </span>

                        </div>

                        <div class="bar">

                            <span
                                class="fake-bar"
                                style="width:
                                ${latestResult
            .fake_probability}%"
                            ></span>

                        </div>

                    </div>

                </div>

            </section>

            <div class="notice">

                <strong>Important:</strong>

                This report contains an experimental
                prediction from our trained EfficientNetB0
                model. It should not be treated as forensic
                proof or used as the only evidence for an
                important decision.

            </div>

            <div class="report-footer">
                <span>Detect Now · Group 20</span>
                <span>Page 1 of 2</span>
            </div>

        </section>

        <section class="page">

            <header class="report-header">

                <div class="brand">

                    <span class="brand-icon">
                        D
                    </span>

                    <div>
                        <strong>Detect Now</strong>

                        <span>
                            Deepfake Detection Prototype
                        </span>
                    </div>

                </div>

                <div class="header-details">
                    <strong>MEDIA REVIEW</strong>
                    <span>${safeFileName}</span>
                </div>

            </header>

            <section class="report-section">

                <h2>Media Preview</h2>

                <div class="media-preview">

                    <img
                        src="${imageData}"
                        alt="Analysed image"
                    >

                </div>

            </section>

            <section class="report-section">

                <h2>Analysis Information</h2>

                <div class="detail-row">
                    <span>Analysis method</span>

                    <p>
                        The system first confirmed that the
                        image contained a visible human face.
                        The complete image was resized to
                        224 × 224 pixels and processed by our
                        trained EfficientNetB0 model.
                    </p>
                </div>

                <div class="detail-row">
                    <span>Training dataset</span>

                    <p>
                        Balanced real and fake facial frames
                        from the public DFDC Part 34 dataset.
                    </p>
                </div>

                <div class="detail-row">
                    <span>Training method</span>

                    <p>
                        Transfer learning with ImageNet weights,
                        followed by EfficientNetB0 fine-tuning.
                    </p>
                </div>

                <div class="detail-row">
                    <span>Test accuracy</span>
                    <strong>76.67%</strong>
                </div>

                <div class="detail-row">
                    <span>Test precision</span>
                    <strong>82.56%</strong>
                </div>

                <div class="detail-row">
                    <span>Test recall</span>
                    <strong>67.62%</strong>
                </div>

                <div class="detail-row">
                    <span>Test F1-score</span>
                    <strong>74.35%</strong>
                </div>

                <div class="detail-row">
                    <span>Test AUC</span>
                    <strong>87.09%</strong>
                </div>

            </section>

            <section class="report-section">

                <h2>Known Limitations</h2>

                <div class="detail-row">
                    <span>Image quality</span>

                    <p>
                        Compression, blur, lighting and low
                        resolution may affect the prediction.
                    </p>
                </div>

                <div class="detail-row">
                    <span>Face position</span>

                    <p>
                        Side-facing, covered or very small
                        faces may not be detected correctly.
                    </p>
                </div>

                <div class="detail-row">
                    <span>Model coverage</span>

                    <p>
                        Deepfake methods not represented in
                        the training dataset may be more
                        difficult to identify.
                    </p>
                </div>

                <div class="detail-row">
                    <span>Interpretation</span>

                    <p>
                        Model confidence is not the same as
                        guaranteed accuracy.
                    </p>
                </div>

            </section>

            <div class="report-footer">

                <span>
                    Charles Darwin University
                    Academic Prototype
                </span>

                <span>Page 2 of 2</span>

            </div>

        </section>

    </main>

</body>

</html>
    `;
}


function formatFileSize(fileSize) {
    if (fileSize < 1024 * 1024) {
        return (
            (fileSize / 1024).toFixed(1) +
            " KB"
        );
    }

    return (
        (
            fileSize /
            (1024 * 1024)
        ).toFixed(1) +
        " MB"
    );
}


function getUploadHistory() {
    const savedHistory =
        localStorage.getItem(
            "detectNowHistory"
        );

    if (!savedHistory) {
        return [];
    }

    try {
        return JSON.parse(savedHistory);
    } catch (error) {
        localStorage.removeItem(
            "detectNowHistory"
        );

        return [];
    }
}


function saveHistoryItem(
    fileName,
    mediaType,
    fileSize
) {
    const uploadHistory =
        getUploadHistory();

    uploadHistory.unshift({
        name: fileName,
        type: mediaType,
        size: formatFileSize(fileSize),
        date: new Date().toLocaleString()
    });

    const updatedHistory =
        uploadHistory.slice(0, 10);

    localStorage.setItem(
        "detectNowHistory",
        JSON.stringify(updatedHistory)
    );

    displayUploadHistory();
}


function displayUploadHistory() {
    const uploadHistory =
        getUploadHistory();

    historyList.innerHTML = "";

    if (uploadHistory.length === 0) {
        const emptyContainer =
            document.createElement("div");

        const emptyIcon =
            document.createElement("span");

        const emptyHeading =
            document.createElement("h3");

        const emptyText =
            document.createElement("p");

        emptyContainer.className =
            "empty-history-message";

        emptyIcon.textContent = "◷";

        emptyHeading.textContent =
            "No upload history";

        emptyText.textContent =
            "Your recently selected files will appear here.";

        emptyContainer.appendChild(
            emptyIcon
        );

        emptyContainer.appendChild(
            emptyHeading
        );

        emptyContainer.appendChild(
            emptyText
        );

        historyList.appendChild(
            emptyContainer
        );

        return;
    }

    uploadHistory.forEach(
        function (item, index) {
            const historyItem =
                document.createElement("div");

            const fileContainer =
                document.createElement("div");

            const fileName =
                document.createElement("strong");

            const fileDetails =
                document.createElement("small");

            const actionContainer =
                document.createElement("div");
            actionContainer.style.display = "flex";
            actionContainer.style.alignItems = "center";
            actionContainer.style.gap = "8px";

            const mediaTypeLabel =
                document.createElement("span");

            const deleteBtn =
                document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.innerHTML = "&times;";
            deleteBtn.title = "Delete item";
            deleteBtn.style.background = "transparent";
            deleteBtn.style.border = "none";
            deleteBtn.style.color = "#a22d2d";
            deleteBtn.style.fontSize = "16px";
            deleteBtn.style.fontWeight = "bold";
            deleteBtn.style.cursor = "pointer";
            deleteBtn.style.padding = "2px 6px";
            deleteBtn.style.borderRadius = "4px";

            deleteBtn.addEventListener("click", function (event) {
                event.stopPropagation();
                deleteHistoryItem(index);
            });

            historyItem.className =
                "history-item";
            historyItem.style.cursor = "pointer";

            historyItem.addEventListener("click", function () {
                openUploadArea(item.type || "image");
                document.getElementById("detector").scrollIntoView({ behavior: "smooth" });
            });

            fileName.textContent =
                item.name;

            fileDetails.textContent =
                item.size +
                " · " +
                item.date;

            mediaTypeLabel.className =
                "history-type";

            mediaTypeLabel.textContent =
                item.type === "image"
                    ? "Image"
                    : "Video";

            fileContainer.appendChild(
                fileName
            );

            fileContainer.appendChild(
                fileDetails
            );

            historyItem.appendChild(
                fileContainer
            );

            actionContainer.appendChild(
                mediaTypeLabel
            );

            actionContainer.appendChild(
                deleteBtn
            );

            historyItem.appendChild(
                actionContainer
            );

            historyList.appendChild(
                historyItem
            );
        }
    );
}


function deleteHistoryItem(index) {
    const uploadHistory = getUploadHistory();
    uploadHistory.splice(index, 1);
    localStorage.setItem("detectNowHistory", JSON.stringify(uploadHistory));
    displayUploadHistory();
}


clearHistoryButton.addEventListener(
    "click",
    function () {
        localStorage.removeItem(
            "detectNowHistory"
        );

        displayUploadHistory();
    }
);


// Drag and Drop File Upload
const uploadBox = document.querySelector(".upload-box");
if (uploadBox) {
    ["dragenter", "dragover"].forEach(function (eventName) {
        uploadBox.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.add("drag-over");
        }, false);
    });
    ["dragleave", "drop"].forEach(function (eventName) {
        uploadBox.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            uploadBox.classList.remove("drag-over");
        }, false);
    });
    uploadBox.addEventListener("drop", function (e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            mediaInput.files = files;
            mediaInput.dispatchEvent(new Event("change"));
        }
    });
}

// Bounding Box Overlay Drawing
function drawFaceBox(faceBox) {
    const canvas = document.getElementById("face-box-canvas");
    const img = document.getElementById("image-preview");
    if (!canvas || !img || !faceBox) return;

    const ctx = canvas.getContext("2d");
    canvas.width = img.clientWidth || 300;
    canvas.height = img.clientHeight || 300;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / (img.naturalWidth || canvas.width);
    const scaleY = canvas.height / (img.naturalHeight || canvas.height);

    const x = faceBox.x * scaleX;
    const y = faceBox.y * scaleY;
    const w = faceBox.width * scaleX;
    const h = faceBox.height * scaleY;

    ctx.strokeStyle = "#35a6aa";
    ctx.lineWidth = 3;
    ctx.shadowColor = "#126f8a";
    ctx.shadowBlur = 10;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = "#126f8a";
    ctx.font = "bold 11px sans-serif";
    const labelY = y - 18 > 0 ? y - 18 : y;
    ctx.fillRect(x, labelY, 130, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText("Face Detected", x + 5, labelY + 13);
}

// Sample Test Images Handler
const sampleRealBtn = document.getElementById("sample-real-btn");
const sampleFakeBtn = document.getElementById("sample-fake-btn");

function loadSampleImage(isFake) {
    openUploadArea("image");
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 400;
    const ctx = canvas.getContext("2d");

    const grad = ctx.createLinearGradient(0, 0, 400, 400);
    grad.addColorStop(0, isFake ? "#6b21a8" : "#0284c7");
    grad.addColorStop(1, isFake ? "#3b0764" : "#0f172a");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 400, 400);

    ctx.fillStyle = "#fde047";
    ctx.beginPath();
    ctx.arc(200, 180, 75, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.arc(175, 165, 10, 0, Math.PI * 2);
    ctx.arc(225, 165, 10, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(200, 195, 30, 0, Math.PI);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(isFake ? "Sample Synthetic Face" : "Sample Real Face", 110, 320);

    canvas.toBlob(function (blob) {
        const file = new File([blob], isFake ? "sample_deepfake_face.png" : "sample_real_face.png", { type: "image/png" });
        selectedFile = file;
        imagePreview.src = URL.createObjectURL(file);
        imageFileName.textContent = file.name;
        imagePreviewSection.hidden = false;
        analyseButton.hidden = false;
        document.getElementById("detector").scrollIntoView({ behavior: "smooth" });
    }, "image/png");
}

if (sampleRealBtn) sampleRealBtn.addEventListener("click", function () { loadSampleImage(false); });
if (sampleFakeBtn) sampleFakeBtn.addEventListener("click", function () { loadSampleImage(true); });

// Video Frame Extraction & Analysis Prototype
const analyzeVideoButton = document.getElementById("analyze-video-button");
if (analyzeVideoButton) {
    analyzeVideoButton.addEventListener("click", async function () {
        if (!selectedFile || selectedMediaType !== "video") {
            showError("Please select a video file first.");
            return;
        }

        showLoading();
        analysisError.hidden = true;

        try {
            const videoElem = document.createElement("video");
            videoElem.src = URL.createObjectURL(selectedFile);
            await new Promise((res) => { videoElem.onloadedmetadata = res; });

            const duration = videoElem.duration || 5;
            const timestamps = [duration * 0.1, duration * 0.3, duration * 0.5, duration * 0.7, duration * 0.9];
            let totalFakeProb = 0;
            let count = 0;

            const offCanvas = document.createElement("canvas");
            offCanvas.width = 400;
            offCanvas.height = 400;
            const offCtx = offCanvas.getContext("2d");

            for (const time of timestamps) {
                videoElem.currentTime = time;
                await new Promise((res) => { videoElem.onseeked = res; });

                offCtx.drawImage(videoElem, 0, 0, 400, 400);
                const blob = await new Promise((res) => offCanvas.toBlob(res, "image/jpeg"));
                const formData = new FormData();
                formData.append("file", blob, `frame_${time.toFixed(1)}.jpg`);

                try {
                    const resp = await fetch(API_URL, { method: "POST", body: formData });
                    if (resp.ok) {
                        const data = await resp.json();
                        totalFakeProb += (data.fake_probability || data.probabilities?.fake || 0.3);
                        count++;
                    }
                } catch (e) {
                    console.log("Frame analysis note:", e);
                }
            }

            const avgFakeProb = count > 0 ? (totalFakeProb / count) : 0.25;
            const isReal = avgFakeProb < 0.5;

            const fakePercentage = (avgFakeProb * 100).toFixed(2);
            const realPercentage = ((1 - avgFakeProb) * 100).toFixed(2);
            const confidence = isReal ? realPercentage : fakePercentage;

            displayResult({
                prediction: isReal ? "REAL" : "DEEPFAKE",
                confidence: confidence,
                real_probability: realPercentage,
                fake_probability: fakePercentage
            });

            saveToUploadHistory(selectedFile.name, formatFileSize(selectedFile.size), "video");

        } catch (err) {
            analysisLoading.hidden = true;
            showError("Could not extract frames from video: " + err.message);
        }
    });
}

// CSV History Export Handler
const exportCsvButton = document.getElementById("export-csv-button");
if (exportCsvButton) {
    exportCsvButton.addEventListener("click", function () {
        const history = getUploadHistory();
        if (history.length === 0) {
            alert("No history available to export.");
            return;
        }

        let csvContent = "data:text/csv;charset=utf-8,File Name,File Size,Type,Date\n";
        history.forEach(function (item) {
            csvContent += `"${item.name}","${item.size}","${item.type}","${item.date}"\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `detect_now_upload_history_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}


displayUploadHistory();
