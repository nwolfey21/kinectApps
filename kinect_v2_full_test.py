import sys
import numpy as np
import cv2

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


def draw_body_bone(joints, jointPoints, joint0, joint1, image):
    """
    Draw a bone between two joints.
    """
    joint0State = joints[joint0].TrackingState
    joint1State = joints[joint1].TrackingState

    # Only draw if both joints are tracked or inferred
    if (joint0State == PyKinectV2.TrackingState_NotTracked or
        joint1State == PyKinectV2.TrackingState_NotTracked):
        return

    pt0 = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
    pt1 = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))
    
    # Check if points are within image bounds
    h, w = image.shape[:2]
    if (0 <= pt0[0] < w and 0 <= pt0[1] < h and 
        0 <= pt1[0] < w and 0 <= pt1[1] < h):
        cv2.line(image, pt0, pt1, (0, 255, 0), 2)


def draw_body(joints, jointPoints, image):
    """
    Draw all Kinect bones.
    Based on Microsoft Kinect SDK bone list.
    """

    _bones = [
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
    for bone in _bones:
        draw_body_bone(joints, jointPoints, bone[0], bone[1], image)
    
    # Draw joints as circles for visibility
    h, w = image.shape[:2]
    for i in range(len(jointPoints)):
        jointState = joints[i].TrackingState
        if jointState != PyKinectV2.TrackingState_NotTracked:
            pt = (int(jointPoints[i].x), int(jointPoints[i].y))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                # Green for tracked, yellow for inferred
                color = (0, 255, 0) if jointState == PyKinectV2.TrackingState_Tracked else (0, 255, 255)
                cv2.circle(image, pt, 5, color, -1)


def main():
    print("Initializing Kinect v2...")

    try:
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth |
            PyKinectV2.FrameSourceTypes_Body
        )
    except Exception as e:
        print("ERROR: Unable to initialize Kinect v2:")
        print(e)
        sys.exit(1)

    print("Kinect initialized successfully!")
    print("Press ESC to quit.")

    bodies = None  # Keep last body frame until new one arrives

    while True:
        # Handle body frame first so joints are ready to draw
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()

        # COLOR frame
        color_img = None
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_img = color_frame.reshape(
                (kinect.color_frame_desc.Height,
                 kinect.color_frame_desc.Width, 4)
            ).astype(np.uint8)

            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)

        if color_img is None:
            continue

        # Draw tracked skeletons
        if bodies:
            for i in range(0, kinect.max_body_count):
                try:
                    body = bodies.bodies[i]
                except (IndexError, AttributeError):
                    continue

                if not body.is_tracked:
                    continue
                
                joints = body.joints

                # Map joint positions to color space
                try:
                    jointPoints = kinect.body_joints_to_color_space(joints)
                    draw_body(joints, jointPoints, color_img)
                except Exception:
                    continue

        # DEPTH frame window
        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_img = depth_frame.reshape(
                (kinect.depth_frame_desc.Height,
                 kinect.depth_frame_desc.Width)
            ).astype(np.uint16)

            depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.03)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Kinect Depth Frame", depth_vis)

        cv2.imshow("Kinect Color + Skeleton", color_img)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

