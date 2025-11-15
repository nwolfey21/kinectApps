import sys
import numpy as np
import cv2
import random
import math
import time

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


class PlayerRing:
    """Represents a colored ring assigned to a player."""
    def __init__(self, x, y, color, player_id, radius=40):
        self.x = x
        self.y = y
        self.color = color  # BGR color tuple
        self.player_id = player_id
        self.radius = radius
        self.active = True  # Ring is still in the game
        self.has_hand = False  # Currently has a hand on it
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.time_off_ring = 0.0  # Time hand has been off ring
        self.eliminated = False
        self.rotation = 0.0


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


def check_hand_on_ring(hand_pos, ring, threshold=60):
    """Check if hand is on the ring (within threshold distance)."""
    if not hand_pos or ring.eliminated or not ring.active:
        return False
    
    dx = hand_pos[0] - ring.x
    dy = hand_pos[1] - ring.y
    distance = math.sqrt(dx * dx + dy * dy)
    
    return distance <= (ring.radius + threshold)


def draw_colored_ring(image, ring, time_elapsed=0.0, highlight=False):
    """Draw a colored ring with animation."""
    if not ring.active or ring.eliminated:
        return
    
    x, y = int(ring.x), int(ring.y)
    h, w = image.shape[:2]
    
    if not (0 <= x < w and 0 <= y < h):
        return
    
    # Update rotation
    ring.rotation += 0.1
    
    # Pulse effect if hand is on it
    if ring.has_hand:
        pulse = 1.0 + 0.15 * math.sin(time_elapsed * 3)
    else:
        pulse = 1.0
    
    radius = int(ring.radius * pulse)
    
    # Highlight winner with thicker, brighter ring
    if highlight:
        thickness = 6
        bright_color = tuple(min(255, c + 50) for c in ring.color)
    else:
        thickness = 4
        bright_color = ring.color
    
    # Draw outer ring
    cv2.circle(image, (x, y), radius, ring.color, thickness)
    cv2.circle(image, (x, y), radius - 2, bright_color, 2)
    
    # Draw inner ring
    inner_radius = int(radius * 0.6)
    cv2.circle(image, (x, y), inner_radius, ring.color, 2)
    
    # Add sparkle effect
    sparkle_angle = ring.rotation
    for i in range(4):
        angle = sparkle_angle + i * math.pi / 2
        sparkle_x = int(x + radius * 0.8 * math.cos(angle))
        sparkle_y = int(y + radius * 0.8 * math.sin(angle))
        if 0 <= sparkle_x < w and 0 <= sparkle_y < h:
            cv2.circle(image, (sparkle_x, sparkle_y), 3, bright_color, -1)


def draw_shadow_bone(joints, jointPoints, joint0, joint1, image, canvas_width, canvas_height):
    """Draw a bone between two joints for the body shadow."""
    joint0State = joints[joint0].TrackingState
    joint1State = joints[joint1].TrackingState

    if (joint0State == PyKinectV2.TrackingState_NotTracked or
        joint1State == PyKinectV2.TrackingState_NotTracked):
        return

    pt0_kinect = jointPoints[joint0]
    pt1_kinect = jointPoints[joint1]
    
    if (np.isnan(pt0_kinect.x) or np.isnan(pt0_kinect.y) or not np.isfinite(pt0_kinect.x) or not np.isfinite(pt0_kinect.y) or
        np.isnan(pt1_kinect.x) or np.isnan(pt1_kinect.y) or not np.isfinite(pt1_kinect.x) or not np.isfinite(pt1_kinect.y)):
        return
    
    pt0 = (int(pt0_kinect.x), int(pt0_kinect.y))
    pt1 = (int(pt1_kinect.x), int(pt1_kinect.y))
    
    h, w = image.shape[:2]
    if (0 <= pt0[0] < w and 0 <= pt0[1] < h and 
        0 <= pt1[0] < w and 0 <= pt1[1] < h):
        shadow_color = (40, 40, 40)
        cv2.line(image, pt0, pt1, shadow_color, 3)


def draw_body_shadow(joints, jointPoints, image, canvas_width, canvas_height):
    """Draw a shadow representation of the body skeleton."""
    bones = [
        (PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck),
        (PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid),
        (PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight),
        (PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft),
        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight),
        (PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft),
        (PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight),
        (PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight),
        (PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight),
        (PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft),
        (PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft),
        (PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft),
        (PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight),
        (PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight),
        (PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight),
        (PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft),
        (PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft),
        (PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft),
    ]

    for bone in bones:
        draw_shadow_bone(joints, jointPoints, bone[0], bone[1], image, canvas_width, canvas_height)
    
    h, w = image.shape[:2]
    shadow_joint_color = (30, 30, 30)
    for i in range(len(jointPoints)):
        jointState = joints[i].TrackingState
        if jointState != PyKinectV2.TrackingState_NotTracked:
            pt_kinect = jointPoints[i]
            if not (np.isnan(pt_kinect.x) or np.isnan(pt_kinect.y) or not np.isfinite(pt_kinect.x) or not np.isfinite(pt_kinect.y)):
                pt = (int(pt_kinect.x), int(pt_kinect.y))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(image, pt, 6, shadow_joint_color, -1)


def draw_text_centered(image, text, y_offset=0, font_scale=2.0, thickness=4, color=(0, 255, 255)):
    """Draw centered text on the image."""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - text_width) // 2
    y = h // 2 + y_offset
    
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)


def draw_countdown(image, countdown_value, canvas_width, canvas_height):
    """Draw countdown number in the center."""
    countdown_text = str(countdown_value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 8.0
    thickness = 12
    color = (0, 255, 255)
    
    (text_width, text_height), baseline = cv2.getTextSize(countdown_text, font, font_scale, thickness)
    x = (canvas_width - text_width) // 2
    y = (canvas_height + text_height) // 2
    
    cv2.putText(image, countdown_text, (x, y), font, font_scale, (0, 0, 0), thickness + 4)
    cv2.putText(image, countdown_text, (x, y), font, font_scale, color, thickness)


def main():
    print("Initializing Kinect v2 for Hand on Ring Game...")

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

    # Canvas dimensions match Kinect color frame
    canvas_width = kinect.color_frame_desc.Width
    canvas_height = kinect.color_frame_desc.Height
    print(f"Canvas size: {canvas_width}x{canvas_height}")

    # Player colors (BGR format)
    player_colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
    ]

    # Game state management
    game_state = 'start_countdown'  # 'start_countdown', 'playing', 'winner', 'new_round_countdown'
    start_countdown_start_time = None
    game_start_time = None
    winner_start_time = None
    new_round_countdown_start_time = None
    
    # Single-player challenge mode
    single_player_mode = False
    challenge_duration = 20.0  # 20 seconds for single player challenge
    
    # Player and ring tracking
    player_rings = {}  # body_id -> PlayerRing
    active_players = set()
    winner_id = None
    
    bodies = None
    current_time = time.time()
    start_countdown_start_time = current_time
    last_frame_time = current_time

    def reset_game():
        """Reset game state for a new round."""
        nonlocal player_rings, active_players, winner_id, game_state
        nonlocal start_countdown_start_time, game_start_time, winner_start_time
        nonlocal new_round_countdown_start_time, last_frame_time, single_player_mode
        
        player_rings = {}
        active_players = set()
        winner_id = None
        game_state = 'start_countdown'
        start_countdown_start_time = time.time()
        game_start_time = None
        winner_start_time = None
        new_round_countdown_start_time = None
        last_frame_time = time.time()
        single_player_mode = False

    def initialize_rings_for_players():
        """Initialize rings for all tracked players."""
        nonlocal player_rings
        
        player_rings = {}
        ring_radius = 40
        margin = ring_radius + 50
        
        # Avoid outer 10% on left and right
        x_min = int(canvas_width * 0.1) + margin
        x_max = int(canvas_width * 0.9) - margin
        y_min = margin
        y_max = canvas_height - margin
        
        # Place rings in a grid pattern
        num_colors = len(player_colors)
        placed_positions = []
        
        for body_id in active_players:
            if body_id >= num_colors:
                continue  # Skip if too many players
            
            # Try to place ring
            attempts = 0
            placed = False
            while not placed and attempts < 100:
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                
                # Check spacing
                too_close = False
                for px, py in placed_positions:
                    distance = math.sqrt((x - px)**2 + (y - py)**2)
                    if distance < ring_radius * 3:
                        too_close = True
                        break
                
                if not too_close:
                    color = player_colors[body_id % num_colors]
                    player_rings[body_id] = PlayerRing(x, y, color, body_id, ring_radius)
                    placed_positions.append((x, y))
                    placed = True
                
                attempts += 1
            
            if not placed:
                # Place anyway if couldn't find good spot
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                color = player_colors[body_id % num_colors]
                player_rings[body_id] = PlayerRing(x, y, color, body_id, ring_radius)

    while True:
        current_time = time.time()
        
        # Handle body frame
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()

        # Get color frame from Kinect
        color_img = None
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_img = color_frame.reshape(
                (kinect.color_frame_desc.Height,
                 kinect.color_frame_desc.Width, 4)
            ).astype(np.uint8)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
        
        if color_img is not None:
            canvas = color_img.copy()
        else:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Update active players
        if bodies:
            new_active_players = set()
            for i in range(0, kinect.max_body_count):
                try:
                    body = bodies.bodies[i]
                    if body.is_tracked:
                        new_active_players.add(i)
                except (IndexError, AttributeError):
                    continue
            
            # Initialize rings if we're in start_countdown and have players
            if game_state == 'start_countdown':
                if new_active_players != active_players or len(player_rings) == 0:
                    active_players = new_active_players
                    if len(active_players) > 0:
                        initialize_rings_for_players()
                else:
                    active_players = new_active_players
            else:
                active_players = new_active_players

        # Game state machine
        if game_state == 'start_countdown':
            countdown_elapsed = current_time - start_countdown_start_time
            countdown_value = 10 - int(countdown_elapsed)
            
            if countdown_value <= 0:
                # Start the game
                game_state = 'playing'
                game_start_time = current_time
                last_frame_time = current_time
                # Only keep rings for players who have hands on them
                rings_to_remove = []
                if bodies:
                    for body_id, ring in player_rings.items():
                        # Check if this player has a hand on their ring
                        has_hand = False
                        for i in range(0, kinect.max_body_count):
                            try:
                                body = bodies.bodies[i]
                                if body.is_tracked and i == body_id:
                                    joints = body.joints
                                    try:
                                        jointPoints = kinect.body_joints_to_color_space(joints)
                                        left_hand, right_hand = get_hand_positions(joints, jointPoints)
                                        
                                        for hand_pos in [left_hand, right_hand]:
                                            if hand_pos and check_hand_on_ring(hand_pos, ring):
                                                has_hand = True
                                                break
                                    except Exception:
                                        pass
                                    break
                            except (IndexError, AttributeError):
                                continue
                        
                        if not has_hand:
                            ring.active = False
                            ring.eliminated = True
                            rings_to_remove.append(body_id)
                    
                    # Remove eliminated rings
                    for body_id in rings_to_remove:
                        if body_id in player_rings:
                            del player_rings[body_id]
                
                # Check if single player mode
                if len(player_rings) == 1:
                    single_player_mode = True
                    print("Single player challenge mode: Must keep hand on ring for 20 seconds!")
                else:
                    single_player_mode = False
                
                # Initialize velocities for active rings (both modes)
                for ring in player_rings.values():
                    angle = random.uniform(0, 2 * math.pi)
                    speed = 50.0  # Initial speed
                    ring.velocity_x = math.cos(angle) * speed
                    ring.velocity_y = math.sin(angle) * speed
            else:
                # Draw countdown and instruction
                draw_countdown(canvas, countdown_value, canvas_width, canvas_height)
                draw_text_centered(canvas, "PUT YOUR HAND ON YOUR COLORED RING!", y_offset=-100, font_scale=1.5)
                
                # Draw rings and body shadows
                if bodies:
                    body_shadows = []
                    for i in range(0, kinect.max_body_count):
                        try:
                            body = bodies.bodies[i]
                            if body.is_tracked:
                                joints = body.joints
                                try:
                                    jointPoints = kinect.body_joints_to_color_space(joints)
                                    body_shadows.append((joints, jointPoints))
                                    
                                    # Check hand on ring
                                    if i in player_rings:
                                        left_hand, right_hand = get_hand_positions(joints, jointPoints)
                                        ring = player_rings[i]
                                        ring.has_hand = False
                                        for hand_pos in [left_hand, right_hand]:
                                            if hand_pos and check_hand_on_ring(hand_pos, ring):
                                                ring.has_hand = True
                                                break
                                except Exception:
                                    continue
                        except (IndexError, AttributeError):
                            continue
                    
                    for joints, jointPoints in body_shadows:
                        draw_body_shadow(joints, jointPoints, canvas, canvas_width, canvas_height)
                
                # Draw all rings
                for ring in player_rings.values():
                    draw_colored_ring(canvas, ring, countdown_elapsed)

        elif game_state == 'playing':
            game_elapsed = current_time - game_start_time
            
            # Check single-player challenge timeout
            if single_player_mode:
                if game_elapsed >= challenge_duration:
                    # Single player completed the challenge!
                    if len(player_rings) == 1:
                        winner_id = list(player_rings.keys())[0]
                        game_state = 'winner'
                        winner_start_time = current_time
                        print(f"Player {winner_id} completed the 20-second challenge!")
                    continue
                elif len(player_rings) == 0:
                    # Single player lost (hand left ring)
                    print("Single player challenge failed!")
                    game_state = 'new_round_countdown'
                    new_round_countdown_start_time = current_time
                    continue
            
            # Calculate actual frame time
            dt = current_time - last_frame_time
            dt = min(dt, 0.1)  # Cap at 100ms to prevent large jumps
            last_frame_time = current_time
            
            # Update ring positions and check hand tracking
            if bodies:
                
                for i in range(0, kinect.max_body_count):
                    try:
                        body = bodies.bodies[i]
                        if not body.is_tracked or i not in player_rings:
                            continue
                        
                        ring = player_rings[i]
                        if not ring.active or ring.eliminated:
                            continue
                        
                        joints = body.joints
                        try:
                            jointPoints = kinect.body_joints_to_color_space(joints)
                            left_hand, right_hand = get_hand_positions(joints, jointPoints)
                            
                            # Check if hand is on ring
                            ring.has_hand = False
                            for hand_pos in [left_hand, right_hand]:
                                if hand_pos and check_hand_on_ring(hand_pos, ring):
                                    ring.has_hand = True
                                    ring.time_off_ring = 0.0
                                    break
                            
                            if not ring.has_hand:
                                ring.time_off_ring += dt
                                if single_player_mode:
                                    # In single-player mode, any time off ring means loss
                                    if ring.time_off_ring > 0.1:  # Small threshold for single player
                                        ring.active = False
                                        ring.eliminated = True
                                        print(f"Player {i} lost the challenge!")
                                elif ring.time_off_ring >= 2.0:
                                    # Player eliminated (multi-player mode)
                                    ring.active = False
                                    ring.eliminated = True
                                    print(f"Player {i} eliminated!")
                        except Exception:
                            continue
                        
                        # Update ring position if still active (both single and multi-player modes)
                        if ring.active and not ring.eliminated:
                            # Increase speed over time - faster increase in single-player mode
                            if single_player_mode:
                                speed_multiplier = 1.0 + game_elapsed * 0.2  # Faster acceleration for challenge
                            else:
                                speed_multiplier = 1.0 + game_elapsed * 0.1
                            
                            # Update velocity direction randomly occasionally
                            if random.random() < 0.01:  # 1% chance per frame
                                angle = random.uniform(0, 2 * math.pi)
                                base_speed = 50.0 * speed_multiplier
                                ring.velocity_x = math.cos(angle) * base_speed
                                ring.velocity_y = math.sin(angle) * base_speed
                            
                            # Update position
                            ring.x += ring.velocity_x * dt * speed_multiplier
                            ring.y += ring.velocity_y * dt * speed_multiplier
                            
                            # Bounce off walls
                            margin = ring.radius
                            if ring.x < margin or ring.x > canvas_width - margin:
                                ring.velocity_x *= -1
                                ring.x = max(margin, min(ring.x, canvas_width - margin))
                            if ring.y < margin or ring.y > canvas_height - margin:
                                ring.velocity_y *= -1
                                ring.y = max(margin, min(ring.y, canvas_height - margin))
                    except (IndexError, AttributeError):
                        continue
            
            # Check for winner (only in multi-player mode)
            if not single_player_mode:
                active_rings = [r for r in player_rings.values() if r.active and not r.eliminated]
                if len(active_rings) <= 1:
                    if len(active_rings) == 1:
                        winner_id = active_rings[0].player_id
                        game_state = 'winner'
                        winner_start_time = current_time
                        print(f"Player {winner_id} wins!")
                    else:
                        # No winner, reset
                        reset_game()
                        continue
            
            # Draw body shadows
            if bodies:
                body_shadows = []
                for i in range(0, kinect.max_body_count):
                    try:
                        body = bodies.bodies[i]
                        if body.is_tracked:
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
            
            # Draw active rings
            for ring in player_rings.values():
                if ring.active and not ring.eliminated:
                    draw_colored_ring(canvas, ring, game_elapsed)
            
            # Draw timer for single-player challenge
            if single_player_mode:
                remaining_time = challenge_duration - game_elapsed
                if remaining_time > 0:
                    minutes = int(remaining_time // 60)
                    seconds = int(remaining_time % 60)
                    centiseconds = int((remaining_time % 1) * 100)
                    timer_text = f"Challenge: {minutes:02d}:{seconds:02d}.{centiseconds:02d}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2.0
                    thickness = 4
                    color = (0, 255, 255)  # Yellow
                    
                    (text_width, text_height), baseline = cv2.getTextSize(timer_text, font, font_scale, thickness)
                    x = (canvas_width - text_width) // 2
                    y = 100
                    
                    cv2.putText(canvas, timer_text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
                    cv2.putText(canvas, timer_text, (x, y), font, font_scale, color, thickness)
                    
                    # Draw instruction
                    draw_text_centered(canvas, "KEEP YOUR HAND ON THE RING!", y_offset=-150, font_scale=1.5)

        elif game_state == 'winner':
            winner_elapsed = current_time - winner_start_time
            
            if winner_elapsed >= 10.0:
                # Transition to new round countdown
                game_state = 'new_round_countdown'
                new_round_countdown_start_time = current_time
            else:
                # Draw winner celebration
                if winner_id is not None and winner_id in player_rings:
                    winner_ring = player_rings[winner_id]
                    draw_colored_ring(canvas, winner_ring, winner_elapsed, highlight=True)
                    
                    # Draw champion text
                    draw_text_centered(canvas, f"PLAYER {winner_id + 1} WINS!", y_offset=-50, font_scale=3.0, color=winner_ring.color)
                    draw_text_centered(canvas, "CHAMPION!", y_offset=50, font_scale=2.0, color=winner_ring.color)
                
                # Draw body shadows
                if bodies:
                    body_shadows = []
                    for i in range(0, kinect.max_body_count):
                        try:
                            body = bodies.bodies[i]
                            if body.is_tracked:
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

        elif game_state == 'new_round_countdown':
            countdown_elapsed = current_time - new_round_countdown_start_time
            countdown_value = 10 - int(countdown_elapsed)
            
            if countdown_value <= 0:
                # Reset and start new round
                reset_game()
            else:
                # Draw countdown
                draw_countdown(canvas, countdown_value, canvas_width, canvas_height)
                draw_text_centered(canvas, "NEW ROUND STARTING...", y_offset=-100, font_scale=1.5)
                
                # Draw body shadows
                if bodies:
                    body_shadows = []
                    for i in range(0, kinect.max_body_count):
                        try:
                            body = bodies.bodies[i]
                            if body.is_tracked:
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
        cv2.imshow("Kinect Hand on Ring Game", canvas)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
