// Main page sections
const mediaSelectionPage = document.getElementById(
    "media-selection-page"
);

const uploadPage = document.getElementById("upload-page");


// Media selection buttons
const selectImageButton = document.getElementById(
    "select-image-button"
);

const selectVideoButton = document.getElementById(
    "select-video-button"
);

const backButton = document.getElementById("back-button");


// Upload elements
const uploadHeading = document.getElementById("upload-heading");

const uploadDescription = document.getElementById(
    "upload-description"
);

const uploadText = document.getElementById("upload-text");

const fileInformation = document.getElementById(
    "file-information"
);

const mediaInput = document.getElementById("media-input");


// Preview elements
const imagePreviewSection = document.getElementById(
    "image-preview-section"
);

const imagePreview = document.getElementById("image-preview");

const imageFileName = document.getElementById(
    "image-file-name"
);

const videoPreviewSection = document.getElementById(
    "video-preview-section"
);

const videoPreview = document.getElementById("video-preview");

const videoFileName = document.getElementById(
    "video-file-name"
);


// Other elements
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


// Store the selected media type
let selectedMediaType = "";


// Store the current temporary preview address
let currentPreviewURL = "";


// Open the upload section
function openUploadPage(mediaType) {

    selectedMediaType = mediaType;

    mediaSelectionPage.hidden = true;
    uploadPage.hidden = false;

    resetUpload();


    if (mediaType === "image") {

        uploadHeading.textContent = "Upload Image";

        uploadDescription.textContent =
            "Select an image from your device.";

        uploadText.textContent = "Choose an image";

        fileInformation.textContent =
            "JPG, JPEG or PNG · Maximum 8 MB";

        mediaInput.accept =
            ".jpg,.jpeg,.png,image/jpeg,image/png";

    } else {

        uploadHeading.textContent = "Upload Video";

        uploadDescription.textContent =
            "Select a video from your device.";

        uploadText.textContent = "Choose a video";

        fileInformation.textContent =
            "MP4, WebM or MOV · Maximum 50 MB";

        mediaInput.accept =
            ".mp4,.webm,.mov,video/mp4,video/webm,video/quicktime";
    }
}


// Image option
selectImageButton.addEventListener("click", function () {

    openUploadPage("image");

});


// Video option
selectVideoButton.addEventListener("click", function () {

    openUploadPage("video");

});


// Return to media selection
backButton.addEventListener("click", function () {

    resetUpload();

    uploadPage.hidden = true;
    mediaSelectionPage.hidden = false;

});


// Reset the upload area
function resetUpload() {

    mediaInput.value = "";

    imagePreviewSection.hidden = true;
    videoPreviewSection.hidden = true;
    maintenanceMessage.hidden = true;
    uploadAgainButton.hidden = true;

    imagePreview.removeAttribute("src");
    videoPreview.removeAttribute("src");

    imageFileName.textContent = "";
    videoFileName.textContent = "";

    if (currentPreviewURL) {

        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = "";
    }
}


// Process the selected file
mediaInput.addEventListener("change", function () {

    const selectedFile = mediaInput.files[0];

    if (!selectedFile) {
        return;
    }


    // Check image type
    if (
        selectedMediaType === "image" &&
        !selectedFile.type.startsWith("image/")
    ) {

        alert("Please select a valid image.");

        resetUpload();

        return;
    }


    // Check video type
    if (
        selectedMediaType === "video" &&
        !selectedFile.type.startsWith("video/")
    ) {

        alert("Please select a valid video.");

        resetUpload();

        return;
    }


    // Image maximum size: 8 MB
    const imageMaximumSize = 8 * 1024 * 1024;


    // Video maximum size: 50 MB
    const videoMaximumSize = 50 * 1024 * 1024;


    if (
        selectedMediaType === "image" &&
        selectedFile.size > imageMaximumSize
    ) {

        alert("Please select an image smaller than 8 MB.");

        resetUpload();

        return;
    }


    if (
        selectedMediaType === "video" &&
        selectedFile.size > videoMaximumSize
    ) {

        alert("Please select a video smaller than 50 MB.");

        resetUpload();

        return;
    }


    // Create a temporary preview address
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


    // Show the maintenance result
    maintenanceMessage.hidden = false;
    uploadAgainButton.hidden = false;


    // Save the upload in browser history
    saveToHistory(
        selectedFile.name,
        selectedMediaType
    );
});


// Upload another file
uploadAgainButton.addEventListener("click", function () {

    resetUpload();

    // Open the file-selection window again
    mediaInput.click();

});


// Get saved history
function getHistory() {

    const savedHistory = localStorage.getItem(
        "detectNowHistory"
    );

    if (!savedHistory) {
        return [];
    }

    return JSON.parse(savedHistory);
}


// Save a file to history
function saveToHistory(fileName, mediaType) {

    const history = getHistory();

    history.unshift({
        name: fileName,
        type: mediaType,
        date: new Date().toLocaleString()
    });


    // Keep only the latest 10 files
    const updatedHistory = history.slice(0, 10);


    localStorage.setItem(
        "detectNowHistory",
        JSON.stringify(updatedHistory)
    );

    displayHistory();
}


// Display upload history
function displayHistory() {

    const history = getHistory();

    historyList.innerHTML = "";


    if (history.length === 0) {

        const emptyMessage = document.createElement("p");

        emptyMessage.className = "empty-history-message";

        emptyMessage.textContent =
            "No upload history is available.";

        historyList.appendChild(emptyMessage);

        return;
    }


    history.forEach(function (item) {

        const historyItem = document.createElement("div");

        historyItem.className = "history-item";


        const information = document.createElement("div");


        const fileName = document.createElement("strong");

        fileName.textContent = item.name;


        const uploadDate = document.createElement("small");

        uploadDate.textContent = item.date;


        const typeLabel = document.createElement("span");

        typeLabel.className = "history-type";

        typeLabel.textContent =
            item.type === "image" ? "Image" : "Video";


        information.appendChild(fileName);
        information.appendChild(uploadDate);

        historyItem.appendChild(information);
        historyItem.appendChild(typeLabel);

        historyList.appendChild(historyItem);
    });
}


// Clear all history
clearHistoryButton.addEventListener("click", function () {

    localStorage.removeItem("detectNowHistory");

    displayHistory();

});


// Display existing history when the page opens
displayHistory();