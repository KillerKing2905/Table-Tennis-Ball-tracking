import cv2
import numpy as np
from collections import deque

def track_ball(frame, buffer_size=64):
    """
    Track a table tennis ball in the given frame
    
    Args:
        frame: Input video frame
        buffer_size: Size of points buffer for trajectory tracking
    
    Returns:
        processed_frame: Frame with tracking visualization
        ball_data: Dictionary with ball metrics if detected, None otherwise
    """
    # Initialize ball data
    ball_data = None
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for orange/white table tennis ball
    # Adjust these values based on your specific lighting conditions
    lower_bound = np.array([0, 0, 200])  # For white ball
    upper_bound = np.array([180, 70, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply erosion and dilation to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process frame to add visualization
    processed_frame = frame.copy()
    
    # Initialize point buffer for trajectory tracking if not already created
    if not hasattr(track_ball, "pts"):
        track_ball.pts = deque(maxlen=buffer_size)
    
    # If ball contour found
    if contours:
        # Find the largest contour (assumed to be the ball)
        c = max(contours, key=cv2.contourArea)
        
        # Check if the contour is large enough to be a ball
        if cv2.contourArea(c) > 50:
            # Find the minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Calculate ball center using moments
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # Only proceed if the radius is larger than a minimum size
            if radius > 5:
                # Draw circle and centroid
                cv2.circle(processed_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(processed_frame, center, 5, (0, 0, 255), -1)
                
                # Add the centroid to the points queue
                track_ball.pts.appendleft(center)
                
                # Calculate ball metrics (simplified)
                ball_data = {
                    'position': (x, y),
                    'radius': radius,
                    'speed': calculate_speed(track_ball.pts),
                    'spin_type': estimate_spin_type(track_ball.pts)
                }
    
    # Draw trajectory
    for i in range(1, len(track_ball.pts)):
        if track_ball.pts[i - 1] is None or track_ball.pts[i] is None:
            continue
        
        # Thicker lines for more recent motion
        thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
        cv2.line(processed_frame, track_ball.pts[i - 1], track_ball.pts[i], (0, 0, 255), thickness)
    
    return processed_frame, ball_data

def calculate_speed(pts):
    """
    Calculate estimated ball speed based on points history
    This is a simplified calculation and would need calibration
    """
    if len(pts) < 2:
        return 0
        
    # Calculate distance between the most recent points
    # This needs calibration with actual distances for accuracy
    dx = pts[0][0] - pts[1][0]
    dy = pts[0][1] - pts[1][1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Assuming 30 fps, convert to m/s (very rough approximation)
    # Would need proper calibration based on camera and table setup
    return distance * 0.01  # Arbitrary scaling factor

def estimate_spin_type(pts):
    """
    Estimate type of spin based on ball trajectory
    This is a simplified heuristic and would need refinement
    """
    if len(pts) < 10:
        return "Unknown"
    
    # Analyze trajectory curvature
    # This is a simplification and would need more sophisticated algorithms
    x_values = [pt[0] for pt in pts if pt is not None]
    y_values = [pt[1] for pt in pts if pt is not None]
    
    if len(x_values) < 3:
        return "Unknown"
    
    # Very basic heuristic
    x_diff = max(x_values) - min(x_values)
    y_diff = max(y_values) - min(y_values)
    
    if x_diff > y_diff * 2:
        return "Sidespin"
    elif y_diff > x_diff * 2:
        return "Topspin/Backspin"
    else:
        return "Mixed Spin"
