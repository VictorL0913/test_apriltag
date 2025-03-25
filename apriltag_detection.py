import apriltag
import numpy as np
import cv2
import sys

def list_cameras():
    """List available camera devices and their capabilities"""
    print("Checking available camera devices:")
    index = 0
    available_cameras = []
    
    # Try the first 10 camera indices
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Get camera information
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  Camera {index}: {width}x{height}")
            available_cameras.append(index)
        cap.release()
        index += 1
    
    return available_cameras

def main():
    # List and check available cameras
    available_cameras = list_cameras()
    
    if not available_cameras:
        print("Error: No cameras found")
        return
        
    # Ask user to select camera if there are multiple
    camera_index = 0  # Default to first camera
    if len(available_cameras) > 1:
        print("Multiple cameras found. Please select:")
        for i, cam_idx in enumerate(available_cameras):
            print(f"  {i}: Camera {cam_idx}")
        try:
            selection = int(input("Enter selection (0-{}): ".format(len(available_cameras)-1)))
            if 0 <= selection < len(available_cameras):
                camera_index = available_cameras[selection]
        except ValueError:
            print(f"Invalid selection, using default camera {camera_index}")
    
    print(f"Attempting to open camera {camera_index}")
    
    # Initialize the camera with explicit backend for virtual camera support
    cap = cv2.VideoCapture(camera_index)
    
    # Try different backends if default fails
    if not cap.isOpened():
        print(f"Failed to open camera with default backend, trying V4L2...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print(f"Successfully opened camera {camera_index}")
    
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

    frame_count = 0
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            frame_count += 1
            print(f"Failed to grab frame (attempt {frame_count})")
            if frame_count > 5:  # Give up after multiple consecutive failures
                print("Too many failed attempts, exiting")
                break
            continue
        
        frame_count = 0  # Reset counter on successful frame grab

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