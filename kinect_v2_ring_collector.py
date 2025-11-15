import sys
import numpy as np
import cv2
import random
import math
import time

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


class Ring:
    """Represents a collectible gold ring."""
    def __init__(self, x, y, radius=30):
        self.x = x
        self.y = y
        self.radius = radius
        self.collected = False
        self.rotation = 0.0  # For animation
        self.pulse_phase = random.uniform(0, 2 * math.pi)  # Random starting phase for pulsing


def draw_gold_ring(image, ring, time_elapsed=0.0):
    """Draw a gold ring with animation effects."""
    if ring.collected:
        return
    
    x, y = int(ring.x), int(ring.y)
    h, w = image.shape[:2]
    
    # Check bounds - only skip if completely outside canvas
    if not (0 <= x < w and 0 <= y < h):
        return
    
    # Update rotation for spinning effect
    ring.rotation += 0.1
    
    # Pulse effect (size variation)
    pulse = 1.0 + 0.1 * math.sin(time_elapsed * 2 + ring.pulse_phase)
    radius = int(ring.radius * pulse)
    
    # Gold color (BGR format)
    gold_color = (0, 215, 255)  # Gold in BGR
    bright_gold = (0, 255, 255)  # Brighter gold
    
    # Draw outer ring (thick)
    cv2.circle(image, (x, y), radius, gold_color, 4)
    cv2.circle(image, (x, y), radius - 2, bright_gold, 2)
    
    # Draw inner ring
    inner_radius = int(radius * 0.6)
    cv2.circle(image, (x, y), inner_radius, gold_color, 2)
    
    # Add sparkle effect (rotating highlights)
    sparkle_angle = ring.rotation
    for i in range(4):
        angle = sparkle_angle + i * math.pi / 2
        sparkle_x = int(x + radius * 0.8 * math.cos(angle))
        sparkle_y = int(y + radius * 0.8 * math.sin(angle))
        if 0 <= sparkle_x < w and 0 <= sparkle_y < h:
            cv2.circle(image, (sparkle_x, sparkle_y), 3, bright_gold, -1)


def get_hand_positions(joints, jointPoints):
    """Get positions of both hands from Kinect skeleton."""
    hand_left = PyKinectV2.JointType_HandLeft
    hand_right = PyKinectV2.JointType_HandRight
    
    left_pos = None
    right_pos = None
    
    if joints[hand_left].TrackingState != PyKinectV2.TrackingState_NotTracked:
        pt = jointPoints[hand_left]
        if not (np.isnan(pt.x) or np.isnan(pt.y) or not np.isfinite(pt.x) or not np.isfinite(pt.y)):
            left_pos = (pt.x, pt.y)
    
    if joints[hand_right].TrackingState != PyKinectV2.TrackingState_NotTracked:
        pt = jointPoints[hand_right]
        if not (np.isnan(pt.x) or np.isnan(pt.y) or not np.isfinite(pt.x) or not np.isfinite(pt.y)):
            right_pos = (pt.x, pt.y)
    
    return left_pos, right_pos


def map_kinect_to_canvas(kinect_x, kinect_y, canvas_width, canvas_height):
    """Map Kinect color space coordinates to canvas coordinates."""
    # Since canvas matches Kinect color frame dimensions, use coordinates directly
    # Just ensure they're integers and within bounds
    canvas_x = int(kinect_x)
    canvas_y = int(kinect_y)
    
    # Clamp to canvas bounds
    canvas_x = max(0, min(canvas_x, canvas_width - 1))
    canvas_y = max(0, min(canvas_y, canvas_height - 1))
    
    return canvas_x, canvas_y


def check_ring_collision(hand_pos, ring, touch_radius=50):
    """Check if hand position is touching a ring."""
    if ring.collected:
        return False
    
    # Calculate distance between hand and ring center
    dx = hand_pos[0] - ring.x
    dy = hand_pos[1] - ring.y
    distance = math.sqrt(dx * dx + dy * dy)
    
    # Check if within touch radius
    return distance <= (ring.radius + touch_radius)


def draw_score(image, score, total_rings):
    """Draw score on the image."""
    score_text = f"Score: {score}/{total_rings}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 255, 255)  # Yellow
    
    # Get text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(score_text, font, font_scale, thickness)
    
    # Position at top-left with padding
    x = 20
    y = 50
    
    # Draw text with black outline for visibility
    cv2.putText(image, score_text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, score_text, (x, y), font, font_scale, color, thickness)


def draw_timer(image, elapsed_time):
    """Draw timer on the image."""
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    centiseconds = int((elapsed_time % 1) * 100)
    timer_text = f"Time: {minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 255, 255)  # Yellow
    
    # Get text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(timer_text, font, font_scale, thickness)
    
    # Position at top-right with padding
    x = image.shape[1] - text_width - 20
    y = 50
    
    # Draw text with black outline for visibility
    cv2.putText(image, timer_text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, timer_text, (x, y), font, font_scale, color, thickness)


def draw_magnified_results(image, score, total_rings, elapsed_time, canvas_width, canvas_height):
    """Draw magnified score and timer in the center of the screen."""
    # Format time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    centiseconds = int((elapsed_time % 1) * 100)
    timer_text = f"Time: {minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    score_text = f"Score: {score}/{total_rings}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 6
    color = (0, 255, 255)  # Yellow
    
    # Draw score
    (score_width, score_height), baseline = cv2.getTextSize(score_text, font, font_scale, thickness)
    score_x = (canvas_width - score_width) // 2
    score_y = canvas_height // 2 - 50
    
    cv2.putText(image, score_text, (score_x, score_y), font, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(image, score_text, (score_x, score_y), font, font_scale, color, thickness)
    
    # Draw timer
    (timer_width, timer_height), baseline = cv2.getTextSize(timer_text, font, font_scale, thickness)
    timer_x = (canvas_width - timer_width) // 2
    timer_y = canvas_height // 2 + 80
    
    cv2.putText(image, timer_text, (timer_x, timer_y), font, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(image, timer_text, (timer_x, timer_y), font, font_scale, color, thickness)


def draw_countdown(image, countdown_value, canvas_width, canvas_height):
    """Draw countdown number in the center of the screen."""
    countdown_text = str(countdown_value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 8.0
    thickness = 12
    color = (0, 255, 255)  # Yellow
    
    # Get text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(countdown_text, font, font_scale, thickness)
    x = (canvas_width - text_width) // 2
    y = (canvas_height + text_height) // 2
    
    # Draw text with black outline for visibility
    cv2.putText(image, countdown_text, (x, y), font, font_scale, (0, 0, 0), thickness + 4)
    cv2.putText(image, countdown_text, (x, y), font, font_scale, color, thickness)


def draw_hand_indicator(image, hand_pos, canvas_width, canvas_height):
    """Draw a visual indicator at hand position."""
    if hand_pos is None:
        return
    
    canvas_x, canvas_y = map_kinect_to_canvas(hand_pos[0], hand_pos[1], canvas_width, canvas_height)
    
    h, w = image.shape[:2]
    if 0 <= canvas_x < w and 0 <= canvas_y < h:
        # Draw a small circle at hand position
        cv2.circle(image, (canvas_x, canvas_y), 15, (0, 255, 0), 2)
        cv2.circle(image, (canvas_x, canvas_y), 5, (0, 255, 0), -1)


def draw_shadow_bone(joints, jointPoints, joint0, joint1, image, canvas_width, canvas_height):
    """Draw a bone between two joints for the body shadow."""
    joint0State = joints[joint0].TrackingState
    joint1State = joints[joint1].TrackingState

    # Only draw if both joints are tracked or inferred
    if (joint0State == PyKinectV2.TrackingState_NotTracked or
        joint1State == PyKinectV2.TrackingState_NotTracked):
        return

    # Map joints to canvas space
    pt0_kinect = jointPoints[joint0]
    pt1_kinect = jointPoints[joint1]
    
    if (np.isnan(pt0_kinect.x) or np.isnan(pt0_kinect.y) or not np.isfinite(pt0_kinect.x) or not np.isfinite(pt0_kinect.y) or
        np.isnan(pt1_kinect.x) or np.isnan(pt1_kinect.y) or not np.isfinite(pt1_kinect.x) or not np.isfinite(pt1_kinect.y)):
        return
    
    pt0 = map_kinect_to_canvas(pt0_kinect.x, pt0_kinect.y, canvas_width, canvas_height)
    pt1 = map_kinect_to_canvas(pt1_kinect.x, pt1_kinect.y, canvas_width, canvas_height)
    
    # Check if points are within image bounds
    h, w = image.shape[:2]
    if (0 <= pt0[0] < w and 0 <= pt0[1] < h and 
        0 <= pt1[0] < w and 0 <= pt1[1] < h):
        # Draw shadow bone (dark gray, semi-transparent)
        shadow_color = (40, 40, 40)  # Dark gray in BGR
        cv2.line(image, pt0, pt1, shadow_color, 3)


def draw_body_shadow(joints, jointPoints, image, canvas_width, canvas_height):
    """Draw a shadow representation of the body skeleton."""
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
        draw_shadow_bone(joints, jointPoints, bone[0], bone[1], image, canvas_width, canvas_height)
    
    # Draw joints as dark circles for shadow effect
    h, w = image.shape[:2]
    shadow_joint_color = (30, 30, 30)  # Dark gray for joints
    for i in range(len(jointPoints)):
        jointState = joints[i].TrackingState
        if jointState != PyKinectV2.TrackingState_NotTracked:
            pt_kinect = jointPoints[i]
            if not (np.isnan(pt_kinect.x) or np.isnan(pt_kinect.y) or not np.isfinite(pt_kinect.x) or not np.isfinite(pt_kinect.y)):
                pt = map_kinect_to_canvas(pt_kinect.x, pt_kinect.y, canvas_width, canvas_height)
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(image, pt, 6, shadow_joint_color, -1)


def main():
    print("Initializing Kinect v2 for Ring Collector Game...")

    try:
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Body
        )
    except Exception as e:
        print("ERROR: Unable to initialize Kinect v2:")
        print(e)
        sys.exit(1)

    print("Kinect initialized successfully!")
    print("Press ESC to quit.")
    print("Move your hands to collect the gold rings!")

    # Canvas dimensions match Kinect color frame
    canvas_width = kinect.color_frame_desc.Width
    canvas_height = kinect.color_frame_desc.Height
    print(f"Canvas size: {canvas_width}x{canvas_height}")
    
    # Initialize rings (a dozen or so)
    num_rings = 15
    rings = []
    ring_radius = 30
    max_pulse_radius = int(ring_radius * 1.1)  # Account for pulse effect
    margin = max_pulse_radius + 20  # Extra margin for safety
    
    # Avoid last 10% on left and right sides
    x_min = int(canvas_width * 0.1) + margin
    x_max = int(canvas_width * 0.9) - margin
    y_min = margin
    y_max = canvas_height - margin
    
    # Try to place rings with minimum spacing to avoid overlap
    min_spacing = ring_radius * 2 + 20  # Minimum distance between ring centers
    placed_positions = []
    max_attempts = 1000
    
    for ring_idx in range(num_rings):
        attempts = 0
        placed = False
        
        while not placed and attempts < max_attempts:
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            
            # Check if this position is far enough from existing rings
            too_close = False
            for px, py in placed_positions:
                distance = math.sqrt((x - px)**2 + (y - py)**2)
                if distance < min_spacing:
                    too_close = True
                    break
            
                if not too_close:
                    rings.append(Ring(x, y, ring_radius))
                    placed_positions.append((x, y))
                    placed = True
                
                attempts += 1
            
            # If we couldn't place with spacing, just place it anyway
            if not placed:
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                rings.append(Ring(x, y, ring_radius))
                placed_positions.append((x, y))
    
    print(f"Initial game: {len(rings)} rings placed")
    
    # Game state management
    game_state = 'playing'  # 'playing', 'showing_results', 'countdown'
    game_start_time = None
    game_end_time = None
    results_start_time = None
    countdown_start_time = None
    final_time = 0.0
    
    # Score tracking
    score = 0
    total_rings = num_rings
    
    bodies = None
    current_time = time.time()
    game_start_time = current_time

    def reset_game():
        """Reset game state for a new round."""
        nonlocal rings, score, game_state, game_start_time, game_end_time
        nonlocal results_start_time, countdown_start_time, final_time
        
        # Reset rings with better placement to ensure all are visible
        rings = []
        ring_radius = 30
        max_pulse_radius = int(ring_radius * 1.1)  # Account for pulse effect
        margin = max_pulse_radius + 20  # Extra margin for safety
        
        # Avoid last 10% on left and right sides
        x_min = int(canvas_width * 0.1) + margin
        x_max = int(canvas_width * 0.9) - margin
        y_min = margin
        y_max = canvas_height - margin
        
        # Try to place rings with minimum spacing to avoid overlap
        min_spacing = ring_radius * 2 + 20  # Minimum distance between ring centers
        placed_positions = []
        max_attempts = 1000
        
        for ring_idx in range(num_rings):
            attempts = 0
            placed = False
            
            while not placed and attempts < max_attempts:
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                
                # Check if this position is far enough from existing rings
                too_close = False
                for px, py in placed_positions:
                    distance = math.sqrt((x - px)**2 + (y - py)**2)
                    if distance < min_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    rings.append(Ring(x, y, ring_radius))
                    placed_positions.append((x, y))
                    placed = True
                
                attempts += 1
            
            # If we couldn't place with spacing, just place it anyway
            if not placed:
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                rings.append(Ring(x, y, ring_radius))
                placed_positions.append((x, y))
        
        print(f"Game reset: {len(rings)} rings placed")
        
        # Reset score and state
        score = 0
        game_state = 'playing'
        game_start_time = time.time()
        game_end_time = None
        results_start_time = None
        countdown_start_time = None
        final_time = 0.0

    while True:
        current_time = time.time()
        
        # Handle body frame
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()

        # Get color frame from Kinect for mixed reality background
        color_img = None
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_img = color_frame.reshape(
                (kinect.color_frame_desc.Height,
                 kinect.color_frame_desc.Width, 4)
            ).astype(np.uint8)
            # Convert BGRA to BGR for OpenCV
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
        
        # Use color frame as canvas, or create black canvas if no frame available
        if color_img is not None:
            canvas = color_img.copy()
        else:
            # Fallback to black canvas if no color frame available yet
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Game state machine
        if game_state == 'playing':
            # Update timer
            elapsed_time = current_time - game_start_time
            
            # Store body data for shadow drawing
            body_shadows = []
            
            # Process tracked bodies
            if bodies:
                for i in range(0, kinect.max_body_count):
                    try:
                        body = bodies.bodies[i]
                        if not body.is_tracked:
                            continue
                        
                        joints = body.joints
                        
                        try:
                            jointPoints = kinect.body_joints_to_color_space(joints)
                        except Exception:
                            continue
                        
                        # Store joints for shadow drawing
                        body_shadows.append((joints, jointPoints))
                        
                        # Get hand positions
                        left_hand, right_hand = get_hand_positions(joints, jointPoints)
                        
                        # Check collisions for left hand
                        if left_hand:
                            canvas_x, canvas_y = map_kinect_to_canvas(
                                left_hand[0], left_hand[1], 
                                canvas_width, canvas_height
                            )
                            hand_pos_canvas = (canvas_x, canvas_y)
                            
                            # Check collision with each ring
                            for ring in rings:
                                if check_ring_collision(hand_pos_canvas, ring):
                                    if not ring.collected:
                                        ring.collected = True
                                        score += 1
                                        print(f"Ring collected! Score: {score}/{total_rings}")
                                        
                                        # Check if all rings collected
                                        if score >= total_rings:
                                            game_end_time = current_time
                                            final_time = elapsed_time
                                            game_state = 'showing_results'
                                            results_start_time = current_time
                                            print(f"All rings collected! Time: {final_time:.2f} seconds")
                        
                        # Check collisions for right hand
                        if right_hand:
                            canvas_x, canvas_y = map_kinect_to_canvas(
                                right_hand[0], right_hand[1], 
                                canvas_width, canvas_height
                            )
                            hand_pos_canvas = (canvas_x, canvas_y)
                            
                            # Check collision with each ring
                            for ring in rings:
                                if check_ring_collision(hand_pos_canvas, ring):
                                    if not ring.collected:
                                        ring.collected = True
                                        score += 1
                                        print(f"Ring collected! Score: {score}/{total_rings}")
                                        
                                        # Check if all rings collected
                                        if score >= total_rings:
                                            game_end_time = current_time
                                            final_time = elapsed_time
                                            game_state = 'showing_results'
                                            results_start_time = current_time
                                            print(f"All rings collected! Time: {final_time:.2f} seconds")
                        
                        # Draw hand indicators (optional, for debugging)
                        # draw_hand_indicator(canvas, left_hand, canvas_width, canvas_height)
                        # draw_hand_indicator(canvas, right_hand, canvas_width, canvas_height)
                    except (IndexError, AttributeError):
                        continue

            # Draw body shadows
            for joints, jointPoints in body_shadows:
                draw_body_shadow(joints, jointPoints, canvas, canvas_width, canvas_height)

            # Draw all rings
            for ring in rings:
                if not ring.collected:
                    # Always try to draw if ring is not collected
                    draw_gold_ring(canvas, ring, elapsed_time)
            
            # Draw score and timer
            draw_score(canvas, score, total_rings)
            draw_timer(canvas, elapsed_time)
            
        elif game_state == 'showing_results':
            # Show magnified results for 5 seconds
            results_elapsed = current_time - results_start_time
            
            if results_elapsed >= 5.0:
                # Transition to countdown
                game_state = 'countdown'
                countdown_start_time = current_time
            else:
                # Draw magnified results
                draw_magnified_results(canvas, score, total_rings, final_time, canvas_width, canvas_height)
                
                # Still draw body shadows if available
                if bodies:
                    body_shadows = []
                    for i in range(0, kinect.max_body_count):
                        try:
                            body = bodies.bodies[i]
                            if not body.is_tracked:
                                continue
                            joints = body.joints
                            try:
                                jointPoints = kinect.body_joints_to_color_space(joints)
                                body_shadows.append((joints, jointPoints))
                            except Exception:
                                continue
                        except (IndexError, AttributeError):
                            continue
                    
                    for joints, jointPoints in body_shadows:
                        draw_body_shadow(joints, jointPoints, canvas, canvas_width, canvas_height)
            
        elif game_state == 'countdown':
            # Countdown from 10
            countdown_elapsed = current_time - countdown_start_time
            countdown_value = 10 - int(countdown_elapsed)
            
            if countdown_value <= 0:
                # Reset and start new game
                reset_game()
            else:
                # Draw countdown
                draw_countdown(canvas, countdown_value, canvas_width, canvas_height)
                
                # Still draw body shadows if available
                if bodies:
                    body_shadows = []
                    for i in range(0, kinect.max_body_count):
                        try:
                            body = bodies.bodies[i]
                            if not body.is_tracked:
                                continue
                            joints = body.joints
                            try:
                                jointPoints = kinect.body_joints_to_color_space(joints)
                                body_shadows.append((joints, jointPoints))
                            except Exception:
                                continue
                        except (IndexError, AttributeError):
                            continue
                    
                    for joints, jointPoints in body_shadows:
                        draw_body_shadow(joints, jointPoints, canvas, canvas_width, canvas_height)
        
        # Display canvas
        cv2.imshow("Kinect Ring Collector", canvas)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
