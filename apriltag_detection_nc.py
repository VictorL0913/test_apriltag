import cv2
import apriltag

# Load the image
image_path = "image.png"  # Ensure this is the correct path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Initialize the AprilTag detector
detector = apriltag.Detector()

# Detect AprilTags in the image
tags = detector.detect(image)

# Check if any tags were found
if tags:
    print(f"Detected {len(tags)} AprilTags!")
    for tag in tags:
        print(f"Tag ID: {tag.tag_id}, Center: {tag.center}")
else:
    print("No AprilTags detected.")

# Show the image
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
