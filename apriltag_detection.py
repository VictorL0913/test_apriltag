import apriltag
import numpy as np
import cv2

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    # Initialize AprilTag detector
    detector = apriltag.Detector(apriltag.DetectorOptions(
        families='tag36h11',  # Specify the tag family you're using
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=False,
        refine_pose=False,
        debug=False,
        quad_contours=True
    ))

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        results = detector.detect(gray)
        
        # Process each detected tag
        for r in results:
            # Extract tag ID and corners
            tag_id = r.tag_id
            corners = r.corners.astype(np.int32)
            
            # Draw box around tag
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            
            # Calculate center of tag
            center = np.mean(corners, axis=0).astype(int)
            
            # Draw tag ID at center
            cv2.putText(frame, f"ID: {tag_id}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Print detection information
            print(f"Detected tag ID: {tag_id}")
            print(f"Tag center: {center}")
            print(f"Tag corners: {corners}")
            print("-------------------")

        # Display the frame
        cv2.imshow('AprilTag Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()