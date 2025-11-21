import sys
import numpy as np
import cv2
import math
import time
import random

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


class Puck:
    """Represents the hockey puck."""
    def __init__(self, x, y, radius=15):
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.max_speed = 800.0  # Maximum speed in pixels per second
        self.friction = 0.995  # Friction coefficient (higher = less friction)


class Paddle:
    """Represents the player's paddle (body position)."""
    def __init__(self, x, y, radius=40):
        self.x = x
        self.y = y
        self.radius = radius
        self.last_x = x
        self.last_y = y
        self.velocity_x = 0.0
        self.velocity_y = 0.0


def get_body_feet_positions_3d(joints):
    """Get 3D positions of both feet from Kinect skeleton (camera space)."""
    foot_left = PyKinectV2.JointType_FootLeft
    foot_right = PyKinectV2.JointType_FootRight
    
    left_pos = None
    right_pos = None
    
    if joints[foot_left].TrackingState != PyKinectV2.TrackingState_NotTracked:
        pos = joints[foot_left].Position
        if not (np.isnan(pos.x) or np.isnan(pos.y) or np.isnan(pos.z) or 
                not np.isfinite(pos.x) or not np.isfinite(pos.y) or not np.isfinite(pos.z)):
            left_pos = (pos.x, pos.y, pos.z)
    
    if joints[foot_right].TrackingState != PyKinectV2.TrackingState_NotTracked:
        pos = joints[foot_right].Position
        if not (np.isnan(pos.x) or np.isnan(pos.y) or np.isnan(pos.z) or 
                not np.isfinite(pos.x) or not np.isfinite(pos.y) or not np.isfinite(pos.z)):
            right_pos = (pos.x, pos.y, pos.z)
    
    return left_pos, right_pos


def get_spine_base_position_3d(joints):
    """Get 3D position of spine base (torso center) in camera space."""
    spine_base = PyKinectV2.JointType_SpineBase
    
    if joints[spine_base].TrackingState != PyKinectV2.TrackingState_NotTracked:
        pos = joints[spine_base].Position
        if not (np.isnan(pos.x) or np.isnan(pos.y) or np.isnan(pos.z) or 
                not np.isfinite(pos.x) or not np.isfinite(pos.y) or not np.isfinite(pos.z)):
            return (pos.x, pos.y, pos.z)
    
    return None


def map_3d_to_topdown(x_3d, y_3d, z_3d, rink_width, rink_height):
    """Map 3D camera space coordinates to 2D top-down view.
    
    Kinect camera space:
    - X: left (-) to right (+)
    - Y: up (+) to down (-)
    - Z: forward (+) from camera
    
    Top-down view:
    - X: left to right (same as camera X)
    - Y: forward from camera (same as camera Z)
    """
    # Map X directly (left-right)
    # Map Z to Y (forward-backward)
    # Y (vertical) is ignored for top-down view
    
    # Kinect coordinate system: X in meters, typically -2 to +2
    # Z in meters, typically 0.5 to 4.5
    
    # Define playable area in 3D space
    x_min, x_max = -1.5, 1.5  # meters
    z_min, z_max = 0.8, 3.5   # meters (closer to camera = lower in top-down)
    
    # Normalize X
    normalized_x = (x_3d - x_min) / (x_max - x_min)
    normalized_x = max(0, min(1, normalized_x))  # Clamp to [0, 1]
    rink_x = int(normalized_x * rink_width)
    
    # Normalize Z (forward distance)
    normalized_z = (z_3d - z_min) / (z_max - z_min)
    normalized_z = max(0, min(1, normalized_z))  # Clamp to [0, 1]
    # Invert so closer objects appear lower in top-down view
    rink_y = int((1.0 - normalized_z) * rink_height)
    
    return rink_x, rink_y


def depth_to_topdown(depth_frame, depth_width, depth_height, rink_width, rink_height):
    """Convert depth frame to top-down view."""
    # Create top-down view from depth data
    # Use depth to create a bird's eye view
    topdown = np.zeros((rink_height, rink_width, 3), dtype=np.uint8)
    
    # Map depth to top-down coordinates
    # Kinect depth is typically 512x424
    # We'll create a top-down view by mapping depth values to positions
    
    # Scale depth values to create a top-down visualization
    depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.1)
    
    # Create a simple top-down representation
    # Map depth pixels to top-down view (simplified approach)
    center_x = rink_width // 2
    center_y = rink_height // 2
    
    # Draw a simple rink background
    # Ice color (light blue-gray)
    ice_color = (200, 220, 220)
    topdown[:, :] = ice_color
    
    # Draw rink boundaries
    border_color = (100, 100, 100)
    border_thickness = 5
    cv2.rectangle(topdown, (0, 0), (rink_width - 1, rink_height - 1), border_color, border_thickness)
    
    # Draw center line
    center_line_color = (150, 150, 150)
    cv2.line(topdown, (rink_width // 2, 0), (rink_width // 2, rink_height), center_line_color, 2)
    
    # Draw center circle
    cv2.circle(topdown, (rink_width // 2, rink_height // 2), 80, center_line_color, 2)
    
    # Draw goal areas (simplified)
    goal_width = 100
    goal_depth = 50
    goal_color = (120, 120, 120)
    
    # Left goal
    cv2.rectangle(topdown, (0, rink_height // 2 - goal_width // 2), 
                  (goal_depth, rink_height // 2 + goal_width // 2), goal_color, 2)
    # Right goal
    cv2.rectangle(topdown, (rink_width - goal_depth, rink_height // 2 - goal_width // 2), 
                  (rink_width, rink_height // 2 + goal_width // 2), goal_color, 2)
    
    return topdown


def map_depth_to_topdown(depth_x, depth_y, depth_z, depth_width, depth_height, rink_width, rink_height):
    """Map 3D depth coordinates to 2D top-down view."""
    # Convert depth coordinates to top-down view
    # depth_z is the distance from the camera (in mm)
    # For top-down view, we use depth_x and depth_z as our x and y coordinates
    
    # Kinect depth range is typically 0-8000mm
    # Map to rink coordinates
    # Assume person is standing in front of camera
    # Map depth_x to rink_x, depth_z to rink_y
    
    # Normalize depth coordinates
    # depth_x is typically 0-512, depth_y is 0-424
    # We'll use depth_x and depth_z for top-down mapping
    
    # Simple mapping: use depth_x directly scaled to rink width
    # and use depth_z (distance) to position along rink height
    rink_x = int((depth_x / depth_width) * rink_width)
    
    # For depth_z, map distance to rink_y position
    # Closer objects appear lower in top-down view
    # Typical depth range: 500mm (close) to 8000mm (far)
    min_depth = 500
    max_depth = 4000  # Reasonable range for gameplay
    normalized_z = (depth_z - min_depth) / (max_depth - min_depth)
    normalized_z = max(0, min(1, normalized_z))  # Clamp to [0, 1]
    rink_y = int(normalized_z * rink_height)
    
    return rink_x, rink_y


def map_color_to_topdown(color_x, color_y, color_width, color_height, rink_width, rink_height):
    """Map color space coordinates to top-down view coordinates."""
    # For top-down view, we'll use a simplified mapping
    # Map color coordinates to rink coordinates
    # This creates a top-down perspective from the color frame
    
    # Scale and center
    scale_x = rink_width / color_width
    scale_y = rink_height / color_height
    
    rink_x = int(color_x * scale_x)
    rink_y = int(color_y * scale_y)
    
    # Clamp to rink bounds
    rink_x = max(0, min(rink_x, rink_width - 1))
    rink_y = max(0, min(rink_y, rink_height - 1))
    
    return rink_x, rink_y


def check_puck_paddle_collision(puck, paddle):
    """Check if puck collides with paddle."""
    dx = puck.x - paddle.x
    dy = puck.y - paddle.y
    distance = math.sqrt(dx * dx + dy * dy)
    
    return distance <= (puck.radius + paddle.radius)


def handle_puck_paddle_collision(puck, paddle, dt):
    """Handle collision between puck and paddle."""
    # Calculate collision normal
    dx = puck.x - paddle.x
    dy = puck.y - paddle.y
    distance = math.sqrt(dx * dx + dy * dy)
    
    if distance == 0:
        return
    
    # Normalize
    nx = dx / distance
    ny = dy / distance
    
    # Calculate relative velocity
    rel_vx = puck.velocity_x - paddle.velocity_x
    rel_vy = puck.velocity_y - paddle.velocity_y
    
    # Calculate relative velocity along normal
    rel_vel_normal = rel_vx * nx + rel_vy * ny
    
    # Only resolve if moving towards each other
    if rel_vel_normal > 0:
        return
    
    # Separate objects
    overlap = (puck.radius + paddle.radius) - distance
    if overlap > 0:
        puck.x += nx * overlap * 0.5
        puck.y += ny * overlap * 0.5
        paddle.x -= nx * overlap * 0.5
        paddle.y -= ny * overlap * 0.5
    
    # Calculate impulse (elastic collision with some energy transfer)
    restitution = 1.2  # Slightly elastic for better gameplay
    impulse = -rel_vel_normal * restitution
    
    # Apply impulse to puck
    puck.velocity_x += impulse * nx
    puck.velocity_y += impulse * ny
    
    # Add paddle velocity influence
    paddle_influence = 0.3
    puck.velocity_x += paddle.velocity_x * paddle_influence
    puck.velocity_y += paddle.velocity_y * paddle_influence
    
    # Limit speed
    speed = math.sqrt(puck.velocity_x**2 + puck.velocity_y**2)
    if speed > puck.max_speed:
        scale = puck.max_speed / speed
        puck.velocity_x *= scale
        puck.velocity_y *= scale


def draw_puck(image, puck):
    """Draw the hockey puck."""
    x, y = int(puck.x), int(puck.y)
    h, w = image.shape[:2]
    
    if not (0 <= x < w and 0 <= y < h):
        return
    
    # Draw puck (black circle with white border)
    cv2.circle(image, (x, y), puck.radius, (0, 0, 0), -1)  # Black fill
    cv2.circle(image, (x, y), puck.radius, (255, 255, 255), 2)  # White border
    
    # Add a small highlight
    highlight_x = x - puck.radius // 3
    highlight_y = y - puck.radius // 3
    if 0 <= highlight_x < w and 0 <= highlight_y < h:
        cv2.circle(image, (highlight_x, highlight_y), 3, (200, 200, 200), -1)


def draw_paddle(image, paddle):
    """Draw the player's paddle."""
    x, y = int(paddle.x), int(paddle.y)
    h, w = image.shape[:2]
    
    if not (0 <= x < w and 0 <= y < h):
        return
    
    # Draw paddle (blue circle with white border)
    paddle_color = (255, 100, 0)  # Blue in BGR
    cv2.circle(image, (x, y), paddle.radius, paddle_color, -1)  # Blue fill
    cv2.circle(image, (x, y), paddle.radius, (255, 255, 255), 3)  # White border
    
    # Draw direction indicator (small line showing movement direction)
    if paddle.velocity_x != 0 or paddle.velocity_y != 0:
        speed = math.sqrt(paddle.velocity_x**2 + paddle.velocity_y**2)
        if speed > 10:  # Only show if moving
            dir_x = int(paddle.velocity_x / speed * paddle.radius * 0.7)
            dir_y = int(paddle.velocity_y / speed * paddle.radius * 0.7)
            cv2.line(image, (x, y), (x + dir_x, y + dir_y), (255, 255, 255), 2)


def draw_score(image, score_left, score_right):
    """Draw score on the image."""
    score_text = f"{score_left} - {score_right}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 4
    color = (255, 255, 255)  # White
    
    # Get text size for positioning
    (text_width, text_height), baseline = cv2.getTextSize(score_text, font, font_scale, thickness)
    
    # Position at top center
    x = (image.shape[1] - text_width) // 2
    y = 60
    
    # Draw text with black outline for visibility
    cv2.putText(image, score_text, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, score_text, (x, y), font, font_scale, color, thickness)


def main():
    print("Initializing Kinect v2 for Hockey Game...")

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
    print("Move your body to control the paddle and hit the puck!")

    # Rink dimensions (top-down view)
    rink_width = 800
    rink_height = 600
    
    # Initialize puck
    puck = Puck(rink_width // 2, rink_height // 2, radius=15)
    # Give puck initial random velocity
    angle = random.uniform(0, 2 * math.pi)
    speed = 200.0
    puck.velocity_x = math.cos(angle) * speed
    puck.velocity_y = math.sin(angle) * speed
    
    # Initialize paddle (will be updated from body tracking)
    paddle = Paddle(rink_width // 2, rink_height - 100, radius=40)
    paddle.last_x = paddle.x
    paddle.last_y = paddle.y
    
    # Game state
    score_left = 0
    score_right = 0
    last_goal_time = time.time()
    goal_cooldown = 2.0  # Prevent rapid scoring
    
    bodies = None
    last_frame_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - last_frame_time
        dt = min(dt, 0.1)  # Cap dt to prevent large jumps
        last_frame_time = current_time
        
        # Handle body frame
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()
        
        # Get color frame from Kinect for background
        color_img = None
        if kinect.has_new_color_frame():
            color_frame = kinect.get_last_color_frame()
            color_img = color_frame.reshape(
                (kinect.color_frame_desc.Height,
                 kinect.color_frame_desc.Width, 4)
            ).astype(np.uint8)
            # Convert BGRA to BGR for OpenCV
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
        
        # Handle depth frame for top-down view
        depth_frame = None
        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_img = depth_frame.reshape(
                (kinect.depth_frame_desc.Height,
                 kinect.depth_frame_desc.Width)
            ).astype(np.uint16)
        
        # Create background from color frame or black
        if color_img is not None:
            # Resize color frame to match rink dimensions
            background = cv2.resize(color_img, (rink_width, rink_height))
        else:
            # Fallback: black background
            background = np.zeros((rink_height, rink_width, 3), dtype=np.uint8)
        
        # Create transparent rink overlay
        rink_overlay = np.zeros((rink_height, rink_width, 3), dtype=np.uint8)
        ice_color = (200, 220, 220)
        rink_overlay[:, :] = ice_color
        
        # Draw rink boundaries
        border_color = (100, 100, 100)
        cv2.rectangle(rink_overlay, (0, 0), (rink_width - 1, rink_height - 1), border_color, 5)
        
        # Draw center line
        center_line_color = (150, 150, 150)
        cv2.line(rink_overlay, (rink_width // 2, 0), (rink_width // 2, rink_height), center_line_color, 2)
        cv2.circle(rink_overlay, (rink_width // 2, rink_height // 2), 80, center_line_color, 2)
        
        # Draw goal areas
        goal_width = 100
        goal_depth = 50
        goal_color = (120, 120, 120)
        cv2.rectangle(rink_overlay, (0, rink_height // 2 - goal_width // 2), 
                      (goal_depth, rink_height // 2 + goal_width // 2), goal_color, 2)
        cv2.rectangle(rink_overlay, (rink_width - goal_depth, rink_height // 2 - goal_width // 2), 
                      (rink_width, rink_height // 2 + goal_width // 2), goal_color, 2)
        
        # Blend rink overlay with background (transparent effect)
        alpha = 0.4  # Transparency: 0.0 = fully transparent, 1.0 = fully opaque
        rink = cv2.addWeighted(background, 1.0 - alpha, rink_overlay, alpha, 0)
        
        # Update paddle position from body tracking
        if bodies:
            for i in range(0, kinect.max_body_count):
                try:
                    body = bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    
                    joints = body.joints
                    
                    try:
                        # Get 3D positions (camera space) for top-down view
                        left_foot_3d, right_foot_3d = get_body_feet_positions_3d(joints)
                        
                        # Use average of feet positions, or spine base as fallback
                        if left_foot_3d and right_foot_3d:
                            # Average of both feet for stability
                            body_x_3d = (left_foot_3d[0] + right_foot_3d[0]) / 2
                            body_y_3d = (left_foot_3d[1] + right_foot_3d[1]) / 2
                            body_z_3d = (left_foot_3d[2] + right_foot_3d[2]) / 2
                        else:
                            spine_pos_3d = get_spine_base_position_3d(joints)
                            if spine_pos_3d:
                                body_x_3d, body_y_3d, body_z_3d = spine_pos_3d
                            else:
                                continue
                        
                        # Map 3D camera space to top-down rink coordinates
                        rink_x, rink_y = map_3d_to_topdown(
                            body_x_3d, body_y_3d, body_z_3d,
                            rink_width, rink_height
                        )
                        
                        # Clamp to rink bounds
                        rink_x = max(paddle.radius, min(rink_x, rink_width - paddle.radius))
                        rink_y = max(paddle.radius, min(rink_y, rink_height - paddle.radius))
                        
                        # Calculate paddle velocity
                        if paddle.last_x is not None and paddle.last_y is not None:
                            paddle.velocity_x = (rink_x - paddle.last_x) / dt if dt > 0 else 0
                            paddle.velocity_y = (rink_y - paddle.last_y) / dt if dt > 0 else 0
                            # Smooth velocity to reduce jitter
                            paddle.velocity_x = paddle.velocity_x * 0.7
                            paddle.velocity_y = paddle.velocity_y * 0.7
                        
                        paddle.last_x = paddle.x
                        paddle.last_y = paddle.y
                        paddle.x = rink_x
                        paddle.y = rink_y
                        
                        break  # Only track first body
                    except Exception:
                        continue
                except (IndexError, AttributeError):
                    continue
        
        # Update puck physics
        # Apply friction
        puck.velocity_x *= puck.friction
        puck.velocity_y *= puck.friction
        
        # Update position
        puck.x += puck.velocity_x * dt
        puck.y += puck.velocity_y * dt
        
        # Check collision with paddle
        if check_puck_paddle_collision(puck, paddle):
            handle_puck_paddle_collision(puck, paddle, dt)
        
        # Check collision with walls
        margin = puck.radius
        if puck.x < margin:
            puck.x = margin
            puck.velocity_x *= -0.8  # Bounce with some energy loss
        elif puck.x > rink_width - margin:
            puck.x = rink_width - margin
            puck.velocity_x *= -0.8
        
        if puck.y < margin:
            puck.y = margin
            puck.velocity_y *= -0.8
        elif puck.y > rink_height - margin:
            puck.y = rink_height - margin
            puck.velocity_y *= -0.8
        
        # Check for goals
        goal_width = 100
        goal_depth = 50
        goal_y_center = rink_height // 2
        goal_y_top = goal_y_center - goal_width // 2
        goal_y_bottom = goal_y_center + goal_width // 2
        
        time_since_last_goal = current_time - last_goal_time
        
        # Left goal (puck goes into left side)
        if puck.x < goal_depth and goal_y_top <= puck.y <= goal_y_bottom and time_since_last_goal > goal_cooldown:
            score_right += 1
            last_goal_time = current_time
            print(f"Goal! Score: {score_left} - {score_right}")
            # Reset puck to center
            puck.x = rink_width // 2
            puck.y = rink_height // 2
            # Give puck random velocity
            angle = random.uniform(0, 2 * math.pi)
            speed = 200.0
            puck.velocity_x = math.cos(angle) * speed
            puck.velocity_y = math.sin(angle) * speed
        
        # Right goal (puck goes into right side)
        if puck.x > rink_width - goal_depth and goal_y_top <= puck.y <= goal_y_bottom and time_since_last_goal > goal_cooldown:
            score_left += 1
            last_goal_time = current_time
            print(f"Goal! Score: {score_left} - {score_right}")
            # Reset puck to center
            puck.x = rink_width // 2
            puck.y = rink_height // 2
            # Give puck random velocity
            angle = random.uniform(0, 2 * math.pi)
            speed = 200.0
            puck.velocity_x = math.cos(angle) * speed
            puck.velocity_y = math.sin(angle) * speed
        
        # Draw everything
        draw_puck(rink, puck)
        draw_paddle(rink, paddle)
        draw_score(rink, score_left, score_right)
        
        # Display rink
        cv2.imshow("Kinect Hockey Game (Top-Down View)", rink)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
