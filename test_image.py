import cv2
import apriltag
import numpy as np
from tag_detection import AprilTagDetector

def test_image(image_path):
    # Initialize the AprilTag detector
    detector = AprilTagDetector(camera_source=None)  # No camera needed for image testing

    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the image
    tags = detector.detector.detect(gray)

    for tag in tags:
        # Get the tag's ID and corners
        tag_id = tag.tag_id
        corners = tag.corners

        # Calculate the center of the tag
        center = np.mean(corners, axis=0)

        # Estimate pose (using dummy camera parameters)
        rvec, tvec, _ = cv2.solvePnP(
            np.array([[-detector.tag_size / 2, -detector.tag_size / 2, 0],
                       [detector.tag_size / 2, -detector.tag_size / 2, 0],
                       [detector.tag_size / 2, detector.tag_size / 2, 0],
                       [-detector.tag_size / 2, detector.tag_size / 2, 0]], dtype=np.float32),
            corners, detector.camera_matrix, detector.dist_coeffs)

        # Calculate distance to the tag
        distance = np.linalg.norm(tvec)

        # Draw the tag's corners and center on the frame
        cv2.polylines(frame, [np.int32(corners)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

        # Annotate the tag ID on the image with a background rectangle
        text = f"ID: {tag_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = int(center[0]) - text_size[0] // 2
        text_y = int(center[1]) - 10

        # Draw a filled rectangle behind the text
        cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2), 
                      (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)

        # Put the text on the image
        cv2.putText(frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Print the tag ID, position, and distance
        print(f"Detected tag ID: {tag_id}, Position: {center}, Distance: {distance:.2f} meters")

    # Resize the frame for better visibility
    frame_resized = cv2.resize(frame, (800, 600))  # Resize to 800x600 pixels

    # Display the resulting frame
    cv2.imshow('AprilTag Detection', frame_resized)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image('apriltag2.jpg')  
