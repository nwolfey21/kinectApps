import sys
import numpy as np
import cv2

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


def draw_stick_figure_bone(joints, jointPoints, joint0, joint1, image, color, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0):
    """
    Draw a bone between two joints for the stick figure.
    """
    joint0State = joints[joint0].TrackingState
    joint1State = joints[joint1].TrackingState

    # Only draw if both joints are tracked or inferred
    if (joint0State == PyKinectV2.TrackingState_NotTracked or
        joint1State == PyKinectV2.TrackingState_NotTracked):
        return

    # Get joint point coordinates
    x0 = jointPoints[joint0].x * scale_x + offset_x
    y0 = jointPoints[joint0].y * scale_y + offset_y
    x1 = jointPoints[joint1].x * scale_x + offset_x
    y1 = jointPoints[joint1].y * scale_y + offset_y
    
    # Check for NaN or invalid values
    if (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1) or
        not np.isfinite(x0) or not np.isfinite(y0) or 
        not np.isfinite(x1) or not np.isfinite(y1)):
        return
    
    # Scale and offset the joint points
    pt0 = (int(x0), int(y0))
    pt1 = (int(x1), int(y1))
    
    # Check if points are within image bounds
    h, w = image.shape[:2]
    if (0 <= pt0[0] < w and 0 <= pt0[1] < h and 
        0 <= pt1[0] < w and 0 <= pt1[1] < h):
        # Draw stick figure bone with specified color
        cv2.line(image, pt0, pt1, color, 3)


def draw_stick_figure_joint(joints, jointPoints, joint_idx, image, color, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0):
    """
    Draw a joint as a circle for the stick figure.
    """
    jointState = joints[joint_idx].TrackingState
    if jointState == PyKinectV2.TrackingState_NotTracked:
        return
    
    # Get joint point coordinates
    x = jointPoints[joint_idx].x * scale_x + offset_x
    y = jointPoints[joint_idx].y * scale_y + offset_y
    
    # Check for NaN or invalid values
    if np.isnan(x) or np.isnan(y) or not np.isfinite(x) or not np.isfinite(y):
        return
    
    pt = (int(x), int(y))
    
    h, w = image.shape[:2]
    if 0 <= pt[0] < w and 0 <= pt[1] < h:
        # Draw joint as a circle with specified color
        cv2.circle(image, pt, 6, color, -1)
        # Add a dark border for visibility
        cv2.circle(image, pt, 6, (0, 0, 0), 1)


def detect_arm_position(joints, jointPoints):
    """
    Detect arm position: 'up', 'down', or 'wide'
    Returns the most prominent arm position.
    """
    # Check if required joints are tracked
    def is_tracked(joint_idx):
        return joints[joint_idx].TrackingState != PyKinectV2.TrackingState_NotTracked
    
    def get_point(joint_idx):
        pt = jointPoints[joint_idx]
        if np.isnan(pt.x) or np.isnan(pt.y) or not np.isfinite(pt.x) or not np.isfinite(pt.y):
            return None
        return (pt.x, pt.y)
    
    # Get shoulder and hand positions
    shoulder_left = get_point(PyKinectV2.JointType_ShoulderLeft) if is_tracked(PyKinectV2.JointType_ShoulderLeft) else None
    shoulder_right = get_point(PyKinectV2.JointType_ShoulderRight) if is_tracked(PyKinectV2.JointType_ShoulderRight) else None
    hand_left = get_point(PyKinectV2.JointType_HandLeft) if is_tracked(PyKinectV2.JointType_HandLeft) else None
    hand_right = get_point(PyKinectV2.JointType_HandRight) if is_tracked(PyKinectV2.JointType_HandRight) else None
    spine_shoulder = get_point(PyKinectV2.JointType_SpineShoulder) if is_tracked(PyKinectV2.JointType_SpineShoulder) else None
    
    arm_positions = []
    
    # Check left arm
    if shoulder_left and hand_left:
        is_up = hand_left[1] < shoulder_left[1] - 50  # Hand significantly above shoulder
        is_wide = False
        if spine_shoulder:
            horizontal_distance = abs(hand_left[0] - spine_shoulder[0])
            vertical_distance = abs(hand_left[1] - shoulder_left[1])
            # Wide if horizontal distance is large relative to vertical
            is_wide = horizontal_distance > 150 and horizontal_distance > vertical_distance * 1.5
        
        if is_wide:
            arm_positions.append('wide')
        elif is_up:
            arm_positions.append('up')
        elif hand_left[1] > shoulder_left[1] + 20:  # Hand below shoulder
            arm_positions.append('down')
    
    # Check right arm
    if shoulder_right and hand_right:
        is_up = hand_right[1] < shoulder_right[1] - 50  # Hand significantly above shoulder
        is_wide = False
        if spine_shoulder:
            horizontal_distance = abs(hand_right[0] - spine_shoulder[0])
            vertical_distance = abs(hand_right[1] - shoulder_right[1])
            # Wide if horizontal distance is large relative to vertical
            is_wide = horizontal_distance > 150 and horizontal_distance > vertical_distance * 1.5
        
        if is_wide:
            arm_positions.append('wide')
        elif is_up:
            arm_positions.append('up')
        elif hand_right[1] > shoulder_right[1] + 20:  # Hand below shoulder
            arm_positions.append('down')
    
    # Determine expression based on arm positions
    # Priority: wide > up > down
    if 'wide' in arm_positions:
        return 'wide'
    elif 'up' in arm_positions:
        return 'up'
    elif 'down' in arm_positions:
        return 'down'
    else:
        return 'down'  # Default to down


def draw_face(image, head_center, head_radius, expression, color):
    """
    Draw a face with expression on the stick figure.
    expression: 'up' (smile), 'down' (frown), or 'wide' (silly)
    """
    h, w = image.shape[:2]
    cx, cy = int(head_center[0]), int(head_center[1])
    
    # Check bounds
    if not (0 <= cx < w and 0 <= cy < h):
        return
    
    # Draw head circle
    cv2.circle(image, (cx, cy), head_radius, color, 2)
    
    # Draw face based on expression
    if expression == 'up':  # Smiling face
        # Eyes
        eye_y = cy - head_radius // 3
        eye_size = head_radius // 6
        cv2.circle(image, (cx - head_radius // 3, eye_y), eye_size, color, -1)
        cv2.circle(image, (cx + head_radius // 3, eye_y), eye_size, color, -1)
        # Smile (arc)
        smile_y = cy + head_radius // 4
        cv2.ellipse(image, (cx, smile_y), (head_radius // 2, head_radius // 3), 
                   0, 0, 180, color, 2)
    
    elif expression == 'down':  # Frowning face
        # Eyes
        eye_y = cy - head_radius // 3
        eye_size = head_radius // 6
        cv2.circle(image, (cx - head_radius // 3, eye_y), eye_size, color, -1)
        cv2.circle(image, (cx + head_radius // 3, eye_y), eye_size, color, -1)
        # Frown (inverted arc)
        frown_y = cy + head_radius // 3
        cv2.ellipse(image, (cx, frown_y), (head_radius // 2, head_radius // 3), 
                   0, 180, 360, color, 2)
    
    elif expression == 'wide':  # Silly face
        # Wide eyes (larger)
        eye_y = cy - head_radius // 4
        eye_size = head_radius // 4
        cv2.circle(image, (cx - head_radius // 3, eye_y), eye_size, color, -1)
        cv2.circle(image, (cx + head_radius // 3, eye_y), eye_size, color, -1)
        # Silly mouth (wide open circle)
        mouth_y = cy + head_radius // 3
        cv2.ellipse(image, (cx, mouth_y), (head_radius // 3, head_radius // 4), 
                   0, 0, 360, color, 2)
        # Tongue (small circle inside mouth)
        cv2.circle(image, (cx, mouth_y + head_radius // 8), head_radius // 6, color, -1)


def draw_stick_figure(joints, jointPoints, image, color, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0):
    """
    Draw a complete stick figure based on Kinect skeleton data.
    """
    # Define the bone connections for the stick figure
    bones = [
        # Torso
        (PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
        (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
        (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),

        # Right Arm
        (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
        (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
        (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),

        # Left Arm
        (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
        (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
        (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),

        # Right Leg
        (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
        (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
        (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight),

        # Left Leg
        (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
        (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
        (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
    ]

    # Draw all bones
    for bone in bones:
        draw_stick_figure_bone(joints, jointPoints, bone[0], bone[1], image, color, scale_x, scale_y, offset_x, offset_y)
    
    # Draw all joints (except head, we'll draw that separately with face)
    for i in range(len(jointPoints)):
        if i != PyKinectV2.JointType_Head:  # Skip head joint, we'll draw it with the face
            draw_stick_figure_joint(joints, jointPoints, i, image, color, scale_x, scale_y, offset_x, offset_y)
    
    # Detect arm position for facial expression
    arm_position = detect_arm_position(joints, jointPoints)
    
    # Draw head and face
    head_joint = PyKinectV2.JointType_Head
    if joints[head_joint].TrackingState != PyKinectV2.TrackingState_NotTracked:
        head_pt = jointPoints[head_joint]
        head_x = head_pt.x * scale_x + offset_x
        head_y = head_pt.y * scale_y + offset_y
        
        # Check for valid head position
        if not (np.isnan(head_x) or np.isnan(head_y) or not np.isfinite(head_x) or not np.isfinite(head_y)):
            # Calculate head radius based on distance from head to neck
            neck_joint = PyKinectV2.JointType_Neck
            if joints[neck_joint].TrackingState != PyKinectV2.TrackingState_NotTracked:
                neck_pt = jointPoints[neck_joint]
                neck_x = neck_pt.x * scale_x + offset_x
                neck_y = neck_pt.y * scale_y + offset_y
                
                if not (np.isnan(neck_x) or np.isnan(neck_y) or not np.isfinite(neck_x) or not np.isfinite(neck_y)):
                    # Head radius is approximately the distance from head to neck
                    head_radius = int(np.sqrt((head_x - neck_x)**2 + (head_y - neck_y)**2) * 0.8)
                    head_radius = max(15, min(head_radius, 40))  # Clamp between 15 and 40 pixels
                else:
                    head_radius = 20  # Default radius
            else:
                head_radius = 20  # Default radius
            
            # Draw face with expression
            draw_face(image, (head_x, head_y), head_radius, arm_position, color)


def calculate_bounding_box(jointPoints):
    """
    Calculate the bounding box of all tracked joints for a single body.
    """
    valid_points = []
    for pt in jointPoints:
        # Check for valid, finite, non-NaN points
        if (pt.x != 0 or pt.y != 0) and not np.isnan(pt.x) and not np.isnan(pt.y) and np.isfinite(pt.x) and np.isfinite(pt.y):
            valid_points.append((pt.x, pt.y))
    
    if not valid_points:
        return None
    
    min_x = min(p[0] for p in valid_points)
    max_x = max(p[0] for p in valid_points)
    min_y = min(p[1] for p in valid_points)
    max_y = max(p[1] for p in valid_points)
    
    # Check if the bounding box is valid
    if np.isnan(min_x) or np.isnan(max_x) or np.isnan(min_y) or np.isnan(max_y):
        return None
    
    return (min_x, min_y, max_x, max_y)


def calculate_combined_bounding_box(all_bounds):
    """
    Calculate the combined bounding box for all bodies.
    """
    if not all_bounds:
        return None
    
    # Filter out any None or invalid bounds
    valid_bounds = [b for b in all_bounds if b is not None]
    if not valid_bounds:
        return None
    
    min_x = min(b[0] for b in valid_bounds)
    min_y = min(b[1] for b in valid_bounds)
    max_x = max(b[2] for b in valid_bounds)
    max_y = max(b[3] for b in valid_bounds)
    
    # Check if the combined bounding box is valid
    if (np.isnan(min_x) or np.isnan(max_x) or np.isnan(min_y) or np.isnan(max_y) or
        not np.isfinite(min_x) or not np.isfinite(max_x) or 
        not np.isfinite(min_y) or not np.isfinite(max_y)):
        return None
    
    return (min_x, min_y, max_x, max_y)


def main():
    print("Initializing Kinect v2 for Stick Figure Animation...")

    try:
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Body
        )
    except Exception as e:
        print("ERROR: Unable to initialize Kinect v2:")
        print(e)
        sys.exit(1)

    print("Kinect initialized successfully!")
    print("Press ESC to quit.")
    print("Move in front of the Kinect to see your stick figure(s)!")

    # Display window dimensions
    display_width = 800
    display_height = 600
    
    # Create a black canvas for the stick figures
    stick_figure_canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    # Define colors for different people (BGR format for OpenCV)
    # Each person gets a distinct color
    person_colors = [
        (255, 255, 255),    # White - Person 1
        (0, 255, 255),      # Yellow - Person 2
        (255, 0, 255),      # Magenta - Person 3
        (0, 255, 0),        # Green - Person 4
        (255, 0, 0),        # Blue - Person 5
        (0, 165, 255),      # Orange - Person 6
    ]

    bodies = None
    # Store last bounds for each tracked body for smoothing
    last_bounds_dict = {}

    while True:
        # Handle body frame
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()

        if bodies is None:
            # Clear canvas if no body detected
            stick_figure_canvas.fill(0)
            cv2.imshow("Stick Figure Animation", stick_figure_canvas)
            if cv2.waitKey(1) == 27:  # ESC
                break
            continue

        # Collect all tracked bodies
        tracked_bodies = []
        for i in range(0, kinect.max_body_count):
            try:
                body = bodies.bodies[i]
                if body.is_tracked:
                    tracked_bodies.append((i, body))
            except (IndexError, AttributeError):
                continue

        if not tracked_bodies:
            # Clear canvas if no tracked bodies
            stick_figure_canvas.fill(0)
            cv2.imshow("Stick Figure Animation", stick_figure_canvas)
            if cv2.waitKey(1) == 27:  # ESC
                break
            continue

        # Process all tracked bodies
        all_bounds = []
        body_data = []
        
        for body_idx, body in tracked_bodies:
            joints = body.joints

            # Map joint positions to color space
            try:
                jointPoints = kinect.body_joints_to_color_space(joints)
            except Exception:
                continue

            # Calculate bounding box of this skeleton
            bounds = calculate_bounding_box(jointPoints)
            if bounds is None:
                continue

            # Smooth the bounds for this body
            body_key = body_idx
            if body_key not in last_bounds_dict:
                last_bounds_dict[body_key] = bounds
            else:
                # Smooth the bounds to reduce jitter
                alpha = 0.3  # Smoothing factor
                last_bounds_dict[body_key] = (
                    alpha * bounds[0] + (1 - alpha) * last_bounds_dict[body_key][0],
                    alpha * bounds[1] + (1 - alpha) * last_bounds_dict[body_key][1],
                    alpha * bounds[2] + (1 - alpha) * last_bounds_dict[body_key][2],
                    alpha * bounds[3] + (1 - alpha) * last_bounds_dict[body_key][3],
                )

            all_bounds.append(last_bounds_dict[body_key])
            body_data.append({
                'joints': joints,
                'jointPoints': jointPoints,
                'color': person_colors[body_idx % len(person_colors)]
            })

        # Clean up bounds for bodies that are no longer tracked
        tracked_indices = {body_idx for body_idx, _ in tracked_bodies}
        last_bounds_dict = {k: v for k, v in last_bounds_dict.items() if k in tracked_indices}

        if not all_bounds:
            continue

        # Calculate combined bounding box for all bodies
        combined_bounds = calculate_combined_bounding_box(all_bounds)
        if combined_bounds is None:
            continue

        min_x, min_y, max_x, max_y = combined_bounds
        
        # Calculate dimensions
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            continue

        # Calculate scale factors to fit all stick figures in display window
        # Leave some padding
        padding = 50
        scale_x = (display_width - 2 * padding) / width
        scale_y = (display_height - 2 * padding) / height
        
        # Check for invalid scale values
        if np.isnan(scale_x) or np.isnan(scale_y) or not np.isfinite(scale_x) or not np.isfinite(scale_y):
            continue
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Check if scale is valid
        if np.isnan(scale) or not np.isfinite(scale) or scale <= 0:
            continue
        
        # Calculate offset to center all stick figures
        scaled_width = width * scale
        scaled_height = height * scale
        offset_x = (display_width - scaled_width) / 2 - min_x * scale
        offset_y = (display_height - scaled_height) / 2 - min_y * scale
        
        # Check for invalid offset values
        if np.isnan(offset_x) or np.isnan(offset_y) or not np.isfinite(offset_x) or not np.isfinite(offset_y):
            continue

        # Clear the canvas
        stick_figure_canvas.fill(0)
        
        # Draw all stick figures
        for data in body_data:
            draw_stick_figure(
                data['joints'], 
                data['jointPoints'], 
                stick_figure_canvas, 
                data['color'],
                scale, 
                scale, 
                offset_x, 
                offset_y
            )
        
        # Display the stick figures
        cv2.imshow("Stick Figure Animation", stick_figure_canvas)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
