import cv2
import apriltag
import numpy as np

class AprilTagDetector:
    def __init__(self, camera_source='/dev/video0', tag_size=0.067):
        # Initialize the camera
        self.camera = cv2.VideoCapture(camera_source)
        # Initialize the AprilTag detector
        self.detector = apriltag.Detector()
        # Define the size of the AprilTag (in meters)
        self.tag_size = tag_size
        # Camera intrinsic parameters (fx, fy, cx, cy)
        self.camera_matrix = np.array([[640, 0, 320],
                                       [0, 480, 240],
                                       [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def start_detection(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the image
            tags = self.detector.detect(gray)

            for tag in tags:
                # Get the tag's ID and corners
                tag_id = tag.tag_id
                corners = tag.corners

                # Calculate the center of the tag
                center = np.mean(corners, axis=0)

                # Estimate pose
                rvec, tvec, _ = cv2.solvePnP(
                    np.array([[-self.tag_size / 2, -self.tag_size / 2, 0],
                               [self.tag_size / 2, -self.tag_size / 2, 0],
                               [self.tag_size / 2, self.tag_size / 2, 0],
                               [-self.tag_size / 2, self.tag_size / 2, 0]], dtype=np.float32),
                    corners, self.camera_matrix, self.dist_coeffs)

                # Calculate distance to the tag
                distance = np.linalg.norm(tvec)

                # Draw the tag's corners and center on the frame
                cv2.polylines(frame, [np.int32(corners)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

                # Print the tag ID, position, and distance
                print(f"Detected tag ID: {tag_id}, Position: {center}, Distance: {distance:.2f} meters")

            # Display the resulting frame
            cv2.imshow('AprilTag Detection', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close windows
        self.camera.release()
        cv2.destroyAllWindows()