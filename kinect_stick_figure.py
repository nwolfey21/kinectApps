import sys
import numpy as np
import cv2

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


def draw_stick_figure_bone(joints, jointPoints, joint0, joint1, image, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0):
    """
    Draw a bone between two joints for the stick figure.
    """
    joint0State = joints[joint0].TrackingState
    joint1State = joints[joint1].TrackingState

    # Only draw if both joints are tracked or inferred
    if (joint0State == PyKinectV2.TrackingState_NotTracked or
        joint1State == PyKinectV2.TrackingState_NotTracked):
        return

    # Scale and offset the joint points
    pt0 = (int(jointPoints[joint0].x * scale_x + offset_x), 
           int(jointPoints[joint0].y * scale_y + offset_y))
    pt1 = (int(jointPoints[joint1].x * scale_x + offset_x), 
           int(jointPoints[joint1].y * scale_y + offset_y))
    
    # Check if points are within image bounds
    h, w = image.shape[:2]
    if (0 <= pt0[0] < w and 0 <= pt0[1] < h and 
        0 <= pt1[0] < w and 0 <= pt1[1] < h):
        # Draw stick figure bone in white
        cv2.line(image, pt0, pt1, (255, 255, 255), 3)


def draw_stick_figure_joint(joints, jointPoints, joint_idx, image, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0):
    """
    Draw a joint as a circle for the stick figure.
    """
    jointState = joints[joint_idx].TrackingState
    if jointState == PyKinectV2.TrackingState_NotTracked:
        return
    
    pt = (int(jointPoints[joint_idx].x * scale_x + offset_x),
          int(jointPoints[joint_idx].y * scale_y + offset_y))
    
    h, w = image.shape[:2]
    if 0 <= pt[0] < w and 0 <= pt[1] < h:
        # Draw joint as a white circle
        cv2.circle(image, pt, 6, (255, 255, 255), -1)
        # Add a border for visibility
        cv2.circle(image, pt, 6, (0, 0, 0), 1)


def draw_stick_figure(joints, jointPoints, image, scale_x=1.0, scale_y=1.0, offset_x=0, offset_y=0):
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
        draw_stick_figure_bone(joints, jointPoints, bone[0], bone[1], image, scale_x, scale_y, offset_x, offset_y)
    
    # Draw all joints
    for i in range(len(jointPoints)):
        draw_stick_figure_joint(joints, jointPoints, i, image, scale_x, scale_y, offset_x, offset_y)


def calculate_bounding_box(jointPoints):
    """
    Calculate the bounding box of all tracked joints.
    """
    valid_points = []
    for pt in jointPoints:
        if pt.x != 0 or pt.y != 0:  # Valid point
            valid_points.append((pt.x, pt.y))
    
    if not valid_points:
        return None
    
    min_x = min(p[0] for p in valid_points)
    max_x = max(p[0] for p in valid_points)
    min_y = min(p[1] for p in valid_points)
    max_y = max(p[1] for p in valid_points)
    
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
    print("Move in front of the Kinect to see your stick figure!")

    # Display window dimensions
    display_width = 800
    display_height = 600
    
    # Create a black canvas for the stick figure
    stick_figure_canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    bodies = None
    last_bounds = None  # Store last bounding box for smoothing

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

        # Find the first tracked body
        tracked_body = None
        for i in range(0, kinect.max_body_count):
            try:
                body = bodies.bodies[i]
                if body.is_tracked:
                    tracked_body = body
                    break
            except (IndexError, AttributeError):
                continue

        if tracked_body is None:
            # Clear canvas if no tracked body
            stick_figure_canvas.fill(0)
            cv2.imshow("Stick Figure Animation", stick_figure_canvas)
            if cv2.waitKey(1) == 27:  # ESC
                break
            continue

        joints = tracked_body.joints

        # Map joint positions to color space (for reference dimensions)
        try:
            jointPoints = kinect.body_joints_to_color_space(joints)
        except Exception:
            continue

        # Calculate bounding box of the skeleton
        bounds = calculate_bounding_box(jointPoints)
        if bounds is None:
            continue

        # Use last bounds if current is invalid (smoothing)
        if last_bounds is None:
            last_bounds = bounds
        else:
            # Smooth the bounds to reduce jitter
            alpha = 0.3  # Smoothing factor
            last_bounds = (
                alpha * bounds[0] + (1 - alpha) * last_bounds[0],
                alpha * bounds[1] + (1 - alpha) * last_bounds[1],
                alpha * bounds[2] + (1 - alpha) * last_bounds[2],
                alpha * bounds[3] + (1 - alpha) * last_bounds[3],
            )

        min_x, min_y, max_x, max_y = last_bounds
        
        # Calculate dimensions
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            continue

        # Calculate scale factors to fit stick figure in display window
        # Leave some padding
        padding = 50
        scale_x = (display_width - 2 * padding) / width
        scale_y = (display_height - 2 * padding) / height
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(scale_x, scale_y)
        
        # Calculate offset to center the stick figure
        scaled_width = width * scale
        scaled_height = height * scale
        offset_x = (display_width - scaled_width) / 2 - min_x * scale
        offset_y = (display_height - scaled_height) / 2 - min_y * scale

        # Clear the canvas
        stick_figure_canvas.fill(0)
        
        # Draw the stick figure
        draw_stick_figure(joints, jointPoints, stick_figure_canvas, scale, scale, offset_x, offset_y)
        
        # Display the stick figure
        cv2.imshow("Stick Figure Animation", stick_figure_canvas)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
