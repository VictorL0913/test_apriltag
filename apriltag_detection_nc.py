import cv2
import apriltag

# Load the image (Ensure the path is correct)
image_path = "image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image loaded correctly
if image is None:
    print("Error: Could not load image. Check the file path.")
    exit()

# Initialize the AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")  # Adjust the tag family if needed
detector = apriltag.Detector(options)

# Detect AprilTags in the image
tags = detector.detect(image)

# Show detection results
if tags:
    print(f"Detected {len(tags)} AprilTags!")
    for tag in tags:
        print(f"Tag ID: {tag.tag_id}, Center: {tag.center}")
else:
    print("No AprilTags detected.")

# Display the image
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
