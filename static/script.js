$(document).ready(function () {
    // --- Initialization ---
    // Hide sections that should not be visible initially when the page loads
    $('.image-section').hide(); // This section contains the image preview and the predict button
    $('.loader').hide();        // The loading spinner
    $('#result').hide();        // The prediction result text

    // --- Image Upload Preview Function ---
    // This function reads the selected file and displays it in the image preview area
    function readURL(input) {
        // Check if a file is actually selected and it's the first file
        if (input.files && input.files[0]) {
            var reader = new FileReader(); // Create a new FileReader object

            // Define what happens when the file is successfully loaded
            reader.onload = function (e) {
                // Set the background image of the imagePreview div to the loaded file's URL
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                // Hide the preview immediately and then fade it in for a smooth visual effect
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650); // Fade in over 650 milliseconds
            }
            // Read the selected file as a Data URL (a base64 encoded string representing the file's data)
            reader.readAsDataURL(input.files[0]);
        }
    }

    // --- Event Listener for Image File Selection ---
    // This fires whenever the user selects a file using the file input
    $("#imageUpload").change(function () {
        // When a file is selected, show the image section (which includes the preview and predict button)
        $('.image-section').show();
        // Explicitly show the predict button, in case it was hidden after a previous prediction
        $('#btn-predict').show();
        // Clear any previous prediction text and hide the result area
        $('#result').text('');
        $('#result').hide();

        // Call the readURL function to display the newly selected image
        readURL(this);
    });

    // --- Event Listener for Predict Button Click ---
    // This fires when the user clicks the "Predict Traffic Signs" button
    $('#btn-predict').click(function () {
        // Create a FormData object from the form with id 'upload-file'
        // FormData is essential for sending files via AJAX
        var form_data = new FormData($('#upload-file')[0]);

        // --- UI Feedback during Prediction ---
        $(this).hide();        // Hide the predict button to prevent multiple clicks
        $('.loader').show();   // Show the loading animation (e.g., a spinner)

        // Make an AJAX (Asynchronous JavaScript and XML) request to your backend
        $.ajax({
            type: 'POST',      // Use the POST method to send data
            url: '/predict',   // The URL endpoint on your server for prediction
            data: form_data,   // The FormData object containing the image file
            contentType: false, // Essential: Prevents jQuery from setting 'Content-Type' header, letting the browser do it for FormData
            cache: false,      // Prevent caching of the request
            processData: false, // Essential: Prevents jQuery from transforming the data, allowing FormData to send the raw file

            // --- Success Callback ---
            // This function runs if the AJAX request is successful (HTTP status 200 OK)
            success: function (data) {
                // Hide the loader and display the prediction result
                $('.loader').hide();
                $('#result').fadeIn(600); // Fade in the result over 600 milliseconds
                $('#result').text(data);  // Set the text of the result area
                console.log('Prediction Successful!');
                // Re-show the predict button so the user can make another prediction or upload a new image
                $('#btn-predict').show();
            },

            // --- Error Callback (Crucial for Robustness) ---
            // This function runs if the AJAX request fails (e.g., server error, network issue)
            error: function (xhr, status, error) {
                // Hide the loader even if there's an error
                $('.loader').hide();
                $('#result').fadeIn(600); // Fade in the result area

                let errorMessage = 'An error occurred during prediction.';
                // Try to get a more specific error message from the server response if available
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMessage = 'Prediction failed: ' + xhr.responseJSON.message;
                } else if (error) {
                    errorMessage = 'Prediction failed: ' + error;
                }
                $('#result').text(errorMessage); // Display the error message
                console.error("AJAX Error:", status, error, xhr); // Log detailed error to console for debugging
                // Re-show the predict button even on error so the user can retry
                $('#btn-predict').show();
            }
        });
    });

}); // End of document.ready