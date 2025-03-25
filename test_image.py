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

        # Print the tag ID, position, and distance
        print(f"Detected tag ID: {tag_id}, Position: {center}, Distance: {distance:.2f} meters")

    # Display the resulting frame
    cv2.imshow('AprilTag Detection', frame)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image('apriltags.png')  # Replace with the path to your PNG image
