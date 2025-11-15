import sys
import numpy as np
import cv2
import math
import time

from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


class PaintParticle:
    """Represents a single paint particle on the canvas."""
    def __init__(self, x, y, vx, vy, color):
        self.x = x
        self.y = y
        self.vx = vx  # velocity x component
        self.vy = vy  # velocity y component
        self.color = color
        self.life = 1.0  # Life remaining (0.0 to 1.0)
        self.decay_rate = 0.001  # Very slow decay - particles stick around
        self.stuck = False  # Whether particle has stopped moving


class PersonTracker:
    """Tracks a person's position and calculates velocity."""
    def __init__(self, body_id):
        self.body_id = body_id
        self.last_position = None
        self.last_time = None
        self.velocity = (0.0, 0.0)
        self.current_position = None


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB color (BGR format for OpenCV)."""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    # Convert to 0-255 range and BGR format
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def get_rainbow_color(hue_offset=0):
    """Get a rainbow color based on time/hue offset."""
    # Cycle through hue values (0-360)
    hue = (time.time() * 60 + hue_offset) % 360  # 60 degrees per second rotation
    saturation = 1.0
    value = 1.0
    return hsv_to_rgb(hue, saturation, value)


def get_body_center_position(joints, jointPoints):
    """Get the center position of a body (using spine base as reference)."""
    spine_base = PyKinectV2.JointType_SpineBase
    if joints[spine_base].TrackingState == PyKinectV2.TrackingState_NotTracked:
        return None
    
    pt = jointPoints[spine_base]
    if np.isnan(pt.x) or np.isnan(pt.y) or not np.isfinite(pt.x) or not np.isfinite(pt.y):
        return None
    
    return (pt.x, pt.y)


def get_hand_positions(joints, jointPoints):
    """Get positions of both hands."""
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
    # Kinect color space is typically 1920x1080
    # Map to canvas with some padding
    kinect_width = 1920
    kinect_height = 1080
    
    # Scale and center
    scale_x = canvas_width / kinect_width
    scale_y = canvas_height / kinect_height
    scale = min(scale_x, scale_y) * 0.9  # Use 90% to add padding
    
    offset_x = (canvas_width - kinect_width * scale) / 2
    offset_y = (canvas_height - kinect_height * scale) / 2
    
    canvas_x = int(kinect_x * scale + offset_x)
    canvas_y = int(kinect_y * scale + offset_y)
    
    return canvas_x, canvas_y


def main():
    print("Initializing Kinect v2 for Paint Canvas...")

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
    print("Move in front of the Kinect to paint the canvas!")

    # Large canvas dimensions
    canvas_width = 1920
    canvas_height = 1080
    
    # Create canvas (start with white background)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Particle system
    particles = []
    max_particles = 5000  # Maximum number of particles
    
    # Track people
    person_trackers = {}  # body_id -> PersonTracker
    
    # Color offset for each person (so they have different rainbow phases)
    color_offsets = {}
    
    bodies = None
    last_frame_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
        
        # Clamp dt to prevent large jumps
        dt = min(dt, 0.1)
        
        # Handle body frame
        if kinect.has_new_body_frame():
            bodies = kinect.get_last_body_frame()

        if bodies is None:
            # Update and draw particles even without new body data
            # Update particles
            for particle in particles[:]:
                if not particle.stuck:
                    particle.x += particle.vx * dt * 60  # Scale velocity
                    particle.y += particle.vy * dt * 60
                    
                    # Check if particle has stopped moving (stuck to canvas)
                    if abs(particle.vx) < 1 and abs(particle.vy) < 1:
                        particle.stuck = True
                        particle.vx = 0
                        particle.vy = 0
                
                # Only decay if not stuck, and very slowly
                if not particle.stuck:
                    particle.life -= particle.decay_rate
                    particle.life = max(0.1, particle.life)  # Keep minimum opacity
                else:
                    # Stuck particles don't decay
                    particle.life = max(0.3, particle.life)
            
            # Don't clear canvas - let particles accumulate and stick
            # Only fade the canvas very slowly to create persistence
            canvas[:, :, :] = np.clip(canvas.astype(np.float32) * 0.999 + 255 * 0.001, 0, 255).astype(np.uint8)
            
            # Draw particles on canvas
            for particle in particles:
                alpha = particle.life
                color = tuple(int(c * alpha) for c in particle.color)
                x, y = int(particle.x), int(particle.y)
                
                if 0 <= x < canvas_width and 0 <= y < canvas_height:
                    # Draw larger particles
                    particle_size = 12 if particle.stuck else 10
                    cv2.circle(canvas, (x, y), particle_size, color, -1)
                    # Add glow effect for moving particles
                    if not particle.stuck:
                        cv2.circle(canvas, (x, y), particle_size + 3, tuple(int(c * alpha * 0.3) for c in particle.color), 1)
            
            # Display canvas
            cv2.imshow("Kinect Paint Canvas", canvas)
            if cv2.waitKey(1) == 27:  # ESC
                break
            continue

        # Process tracked bodies
        tracked_bodies = []
        for i in range(0, kinect.max_body_count):
            try:
                body = bodies.bodies[i]
                if body.is_tracked:
                    tracked_bodies.append((i, body))
            except (IndexError, AttributeError):
                continue

        # Update person trackers
        active_body_ids = set()
        
        for body_idx, body in tracked_bodies:
            active_body_ids.add(body_idx)
            
            joints = body.joints
            
            try:
                jointPoints = kinect.body_joints_to_color_space(joints)
            except Exception:
                continue
            
            # Initialize tracker if needed
            if body_idx not in person_trackers:
                person_trackers[body_idx] = PersonTracker(body_idx)
                color_offsets[body_idx] = body_idx * 60  # Offset each person's color
            
            tracker = person_trackers[body_idx]
            
            # Get current position (use spine base or average of hands)
            center_pos = get_body_center_position(joints, jointPoints)
            left_hand, right_hand = get_hand_positions(joints, jointPoints)
            
            # Use hand positions if available, otherwise use center
            if left_hand and right_hand:
                # Use average of both hands
                current_pos = ((left_hand[0] + right_hand[0]) / 2, 
                              (left_hand[1] + right_hand[1]) / 2)
            elif left_hand:
                current_pos = left_hand
            elif right_hand:
                current_pos = right_hand
            elif center_pos:
                current_pos = center_pos
            else:
                continue
            
            tracker.current_position = current_pos
            
            # Calculate velocity
            if tracker.last_position is not None and tracker.last_time is not None:
                time_diff = current_time - tracker.last_time
                if time_diff > 0:
                    dx = current_pos[0] - tracker.last_position[0]
                    dy = current_pos[1] - tracker.last_position[1]
                    
                    # Calculate velocity (pixels per second)
                    vx = dx / time_diff
                    vy = dy / time_diff
                    
                    # Smooth velocity to reduce jitter
                    alpha = 0.3
                    tracker.velocity = (
                        alpha * vx + (1 - alpha) * tracker.velocity[0],
                        alpha * vy + (1 - alpha) * tracker.velocity[1]
                    )
            
            tracker.last_position = current_pos
            tracker.last_time = current_time
            
            # Generate paint particles based on movement
            if tracker.velocity[0] != 0 or tracker.velocity[1] != 0:
                # Map position to canvas
                canvas_x, canvas_y = map_kinect_to_canvas(
                    current_pos[0], current_pos[1], 
                    canvas_width, canvas_height
                )
                
                # Calculate velocity magnitude
                vel_magnitude = math.sqrt(tracker.velocity[0]**2 + tracker.velocity[1]**2)
                
                # Generate more particles for faster movement
                num_particles = int(min(vel_magnitude / 10, 5))  # Up to 5 particles per frame
                
                # Get rainbow color for this person
                color = get_rainbow_color(color_offsets[body_idx])
                
                for _ in range(num_particles):
                    # Add some randomness to particle position
                    offset_x = np.random.uniform(-5, 5)
                    offset_y = np.random.uniform(-5, 5)
                    
                    # Map velocity to canvas space
                    canvas_vx = tracker.velocity[0] * (canvas_width / 1920)
                    canvas_vy = tracker.velocity[1] * (canvas_height / 1080)
                    
                    # Add some randomness to velocity
                    canvas_vx += np.random.uniform(-20, 20)
                    canvas_vy += np.random.uniform(-20, 20)
                    
                    # Create particle
                    if len(particles) < max_particles:
                        particle = PaintParticle(
                            canvas_x + offset_x,
                            canvas_y + offset_y,
                            canvas_vx,
                            canvas_vy,
                            color
                        )
                        particles.append(particle)
        
        # Remove trackers for bodies that are no longer tracked
        person_trackers = {k: v for k, v in person_trackers.items() if k in active_body_ids}
        
        # Update particles
        for particle in particles[:]:
            if not particle.stuck:
                # Apply gravity and friction to moving particles
                particle.vy += 50 * dt  # Gravity
                particle.vx *= 0.98  # Friction
                particle.vy *= 0.98
                
                particle.x += particle.vx * dt * 60  # Scale velocity
                particle.y += particle.vy * dt * 60
                
                # Check if particle has stopped moving (stuck to canvas)
                if abs(particle.vx) < 1 and abs(particle.vy) < 1:
                    particle.stuck = True
                    particle.vx = 0
                    particle.vy = 0
            
            # Only decay if not stuck, and very slowly
            if not particle.stuck:
                particle.life -= particle.decay_rate
                particle.life = max(0.1, particle.life)  # Keep minimum opacity
            else:
                # Stuck particles don't decay
                particle.life = max(0.3, particle.life)
        
        # Don't clear canvas - let particles accumulate and stick
        # Only fade the canvas very slowly to create persistence
        canvas[:, :, :] = np.clip(canvas.astype(np.float32) * 0.999 + 255 * 0.001, 0, 255).astype(np.uint8)
        
        # Draw particles on canvas
        for particle in particles:
            alpha = particle.life
            color = tuple(int(c * alpha) for c in particle.color)
            x, y = int(particle.x), int(particle.y)
            
            if 0 <= x < canvas_width and 0 <= y < canvas_height:
                # Draw larger particles
                particle_size = 12 if particle.stuck else 10
                cv2.circle(canvas, (x, y), particle_size, color, -1)
                # Add glow effect for moving particles
                if not particle.stuck:
                    cv2.circle(canvas, (x, y), particle_size + 3, tuple(int(c * alpha * 0.3) for c in particle.color), 1)
        
        # Display canvas
        cv2.imshow("Kinect Paint Canvas", canvas)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
