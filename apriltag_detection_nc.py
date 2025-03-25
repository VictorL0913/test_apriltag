import cv2
import apriltag

# Load the PNG image
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Initialize AprilTag detector
detector = apriltag.Detector()

# Detect AprilTags in the image
tags = detector.detect(image)

# Draw detected tags
for tag in tags:
    for idx in range(len(tag.corners)):
        pt1 = tuple(map(int, tag.corners[idx]))
        pt2 = tuple(map(int, tag.corners[(idx + 1) % 4]))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    # Display tag ID
    cv2.putText(image, f"ID: {tag.tag_id}", (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with detected AprilTags
cv2.imshow("AprilTag Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

