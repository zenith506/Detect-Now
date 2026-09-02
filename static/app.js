const mediaSelection = document.getElementById("media-selection");
const uploadArea = document.getElementById("upload-area");

const selectImageButton = document.getElementById(
    "select-image-button"
);

const selectVideoButton = document.getElementById(
    "select-video-button"
);

const backButton = document.getElementById("back-button");

const selectedTypeIcon = document.getElementById(
    "selected-type-icon"
);

const uploadHeading = document.getElementById(
    "upload-heading"
);

const uploadDescription = document.getElementById(
    "upload-description"
);

const uploadText = document.getElementById("upload-text");

const fileInformation = document.getElementById(
    "file-information"
);

const mediaInput = document.getElementById("media-input");

const imagePreviewSection = document.getElementById(
    "image-preview-section"
);

const imagePreview = document.getElementById(
    "image-preview"
);

const imageFileName = document.getElementById(
    "image-file-name"
);

const videoPreviewSection = document.getElementById(
    "video-preview-section"
);

const videoPreview = document.getElementById(
    "video-preview"
);

const videoFileName = document.getElementById(
    "video-file-name"
);

const maintenanceMessage = document.getElementById(
    "maintenance-message"
);

const uploadAgainButton = document.getElementById(
    "upload-again-button"
);

const clearHistoryButton = document.getElementById(
    "clear-history-button"
);

const historyList = document.getElementById("history-list");

const maintenanceModal = document.getElementById(
    "maintenance-modal"
);

const closeModalButton = document.getElementById(
    "close-modal-button"
);

const modalConfirmButton = document.getElementById(
    "modal-confirm-button"
);

const modalBackground = document.querySelector(
    ".modal-background"
);

const maintenanceButtons = document.querySelectorAll(
    ".maintenance-trigger"
);

let selectedMediaType = "";
let currentPreviewURL = "";

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
            "JPG, JPEG or PNG · Maximum 8 MB";

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

selectImageButton.addEventListener("click", function () {
    openUploadArea("image");
});

selectVideoButton.addEventListener("click", function () {
    openUploadArea("video");
});

backButton.addEventListener("click", function () {
    resetSelectedFile();

    uploadArea.hidden = true;
    mediaSelection.hidden = false;

    selectedMediaType = "";
});

function resetSelectedFile() {
    mediaInput.value = "";

    imagePreviewSection.hidden = true;
    videoPreviewSection.hidden = true;
    maintenanceMessage.hidden = true;
    uploadAgainButton.hidden = true;

    imagePreview.removeAttribute("src");

    videoPreview.pause();
    videoPreview.removeAttribute("src");
    videoPreview.load();

    imageFileName.textContent = "";
    videoFileName.textContent = "";

    if (currentPreviewURL) {
        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = "";
    }
}

mediaInput.addEventListener("change", function () {
    const selectedFile = mediaInput.files[0];

    if (!selectedFile) {
        return;
    }

    if (
        selectedMediaType === "image" &&
        !selectedFile.type.startsWith("image/")
    ) {
        alert(
            "Please select a valid JPG, JPEG or PNG image."
        );

        resetSelectedFile();

        return;
    }

    if (
        selectedMediaType === "video" &&
        !selectedFile.type.startsWith("video/")
    ) {
        alert(
            "Please select a valid MP4, WebM or MOV video."
        );

        resetSelectedFile();

        return;
    }

    const imageMaximumSize = 8 * 1024 * 1024;
    const videoMaximumSize = 50 * 1024 * 1024;

    if (
        selectedMediaType === "image" &&
        selectedFile.size > imageMaximumSize
    ) {
        alert("Please select an image smaller than 8 MB.");

        resetSelectedFile();

        return;
    }

    if (
        selectedMediaType === "video" &&
        selectedFile.size > videoMaximumSize
    ) {
        alert("Please select a video smaller than 50 MB.");

        resetSelectedFile();

        return;
    }

    currentPreviewURL = URL.createObjectURL(selectedFile);

    if (selectedMediaType === "image") {
        imagePreview.src = currentPreviewURL;
        imageFileName.textContent = selectedFile.name;

        imagePreviewSection.hidden = false;
        videoPreviewSection.hidden = true;
    } else {
        videoPreview.src = currentPreviewURL;
        videoFileName.textContent = selectedFile.name;

        videoPreviewSection.hidden = false;
        imagePreviewSection.hidden = true;
    }

    maintenanceMessage.hidden = false;
    uploadAgainButton.hidden = false;

    saveHistoryItem(
        selectedFile.name,
        selectedMediaType,
        selectedFile.size
    );
});

uploadAgainButton.addEventListener("click", function () {
    resetSelectedFile();
    mediaInput.click();
});

function formatFileSize(fileSize) {
    if (fileSize < 1024 * 1024) {
        return (fileSize / 1024).toFixed(1) + " KB";
    }

    return (
        (fileSize / (1024 * 1024)).toFixed(1) + " MB"
    );
}

function getUploadHistory() {
    const savedHistory = localStorage.getItem(
        "detectNowHistory"
    );

    if (!savedHistory) {
        return [];
    }

    try {
        return JSON.parse(savedHistory);
    } catch (error) {
        localStorage.removeItem("detectNowHistory");

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

    const updatedHistory = uploadHistory.slice(0, 10);

    localStorage.setItem(
        "detectNowHistory",
        JSON.stringify(updatedHistory)
    );

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

        emptyText.textContent =
            "Your recently selected files will appear here.";

        emptyContainer.appendChild(emptyIcon);
        emptyContainer.appendChild(emptyHeading);
        emptyContainer.appendChild(emptyText);

        historyList.appendChild(emptyContainer);

        return;
    }

    uploadHistory.forEach(function (item) {
        const historyItem = document.createElement("div");

        const fileInformationContainer =
            document.createElement("div");

        const fileName = document.createElement("strong");
        const fileDetails = document.createElement("small");
        const mediaTypeLabel = document.createElement("span");

        historyItem.className = "history-item";
        fileName.textContent = item.name;

        fileDetails.textContent =
            item.size + " · " + item.date;

        mediaTypeLabel.className = "history-type";

        mediaTypeLabel.textContent =
            item.type === "image" ? "Image" : "Video";

        fileInformationContainer.appendChild(fileName);
        fileInformationContainer.appendChild(fileDetails);

        historyItem.appendChild(fileInformationContainer);
        historyItem.appendChild(mediaTypeLabel);

        historyList.appendChild(historyItem);
    });
}

clearHistoryButton.addEventListener("click", function () {
    const uploadHistory = getUploadHistory();

    if (uploadHistory.length === 0) {
        showMaintenanceModal(
            "No History Available",
            "There is currently no upload history to clear."
        );

        return;
    }

    localStorage.removeItem("detectNowHistory");

    displayUploadHistory();
});

function showMaintenanceModal(
    heading = "Feature Under Maintenance",
    message =
        "This function is included in the Detect Now prototype design, but it is not available yet. It will be developed in a future project stage."
) {
    const modalHeading =
        maintenanceModal.querySelector("h2");

    const modalMessage =
        maintenanceModal.querySelector("p");

    modalHeading.textContent = heading;
    modalMessage.textContent = message;

    maintenanceModal.hidden = false;
    document.body.style.overflow = "hidden";
}

function closeMaintenanceModal() {
    maintenanceModal.hidden = true;
    document.body.style.overflow = "";
}

maintenanceButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        showMaintenanceModal();
    });
});

closeModalButton.addEventListener(
    "click",
    closeMaintenanceModal
);

modalConfirmButton.addEventListener(
    "click",
    closeMaintenanceModal
);

modalBackground.addEventListener(
    "click",
    closeMaintenanceModal
);

document.addEventListener("keydown", function (event) {
    if (
        event.key === "Escape" &&
        !maintenanceModal.hidden
    ) {
        closeMaintenanceModal();
    }
});

displayUploadHistory();