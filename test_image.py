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
        # Get the tag's ID, corners, and family
        tag_id = tag.tag_id
        corners = tag.corners
        tag_family = tag.family  # Get the tag family

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

        # Annotate the tag ID and family on the image
        cv2.putText(frame, f"ID: {tag_id}, Family: {tag_family}", 
                    (int(center[0]), int(center[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Print the tag ID, position, and distance
        print(f"Detected tag ID: {tag_id}, Family: {tag_family}, Position: {center}, Distance: {distance:.2f} meters")

    # Resize the frame for better visibility
    frame_resized = cv2.resize(frame, (800, 600))  # Resize to 800x600 pixels

    # Display the resulting frame
    cv2.imshow('AprilTag Detection', frame_resized)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image('apriltag2.jpg')  
