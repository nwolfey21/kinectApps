# Kinect v2 Applications

A collection of Python applications for capturing and visualizing data from Microsoft Kinect v2, including color frames, depth maps, body skeleton tracking, stick figure animation, interactive paint canvas, mixed reality ring collector game, competitive hand-on-ring challenge game, and top-down ice hockey game.

## Features

- **Color Frame Capture**: Real-time color video feed from Kinect v2
- **Depth Frame Visualization**: Depth map visualization with color mapping
- **Body Skeleton Tracking**: Real-time body pose detection and skeleton overlay on color frames
- **Multi-frame Support**: Simultaneous capture of color, depth, and body tracking data
- **Stick Figure Animation**: Real-time stick figure drawing controlled by Kinect body movement with support for multiple people (up to 6)
- **Paint Canvas**: Interactive fluid paint simulation where hand movements control paint flow with rainbow colors on a white canvas
- **Ring Collector Game**: Mixed reality game where players collect gold rings overlaid on live Kinect video feed, featuring body shadow tracking, timer, and automatic game reset
- **Hand on Ring Game**: Competitive elimination game where players must keep their hands on moving colored rings, with single-player challenge mode and progressive difficulty
- **Ice Hockey Game**: Top-down mixed reality hockey game where your body acts as a paddle to hit a puck, featuring transparent rink overlay, physics-based puck movement, goal scoring, and real-time body tracking

## Requirements

- Microsoft Kinect v2 sensor
- Windows OS (Kinect v2 SDK requires Windows)
- Python 3.8
- Conda (for environment management)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd kinectApps
```

2. Create the conda environment from the provided `environment.yml`:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate kinectv2-test
```

## Usage

### Full Test Application

Run the main test script:
```bash
python kinect_v2_full_test.py
```

The application will:
- Initialize the Kinect v2 sensor
- Display two windows:
  - **Kinect Color + Skeleton**: Color frame with overlaid body skeleton
  - **Kinect Depth Frame**: Depth map visualization
- Press **ESC** to quit

### Stick Figure Animation

Run the stick figure animation script:
```bash
python kinect_stick_figure.py
```

The application will:
- Initialize the Kinect v2 sensor (body tracking only)
- Display a single window with a black background
- Draw stick figure(s) that mirror movements in real-time
- Support multiple people simultaneously (up to 6, Kinect v2 limit)
- Assign distinct colors to each person for easy identification:
  - Person 1: White
  - Person 2: Yellow
  - Person 3: Magenta
  - Person 4: Green
  - Person 5: Blue
  - Person 6: Orange
- Automatically scale and center all stick figures in the display window
- Apply smoothing to reduce jitter in the animation
- Press **ESC** to quit

**Note**: Move in front of the Kinect sensor to see your stick figure(s). Each stick figure will follow the corresponding person's body movements, including arms, legs, and torso. Multiple people can be tracked simultaneously, each with their own colored stick figure.

### Paint Canvas

Run the paint canvas script:
```bash
python kinect_v2_paint_canvas.py
```

The application will:
- Initialize the Kinect v2 sensor (body tracking only)
- Display a large white canvas (1920x1080)
- Generate paint particles based on hand/body movement
- Use rainbow colors that cycle continuously
- Particles are larger (10-12 pixels) and stick to the canvas when they stop moving
- Movement speed and direction control particle velocity
- Particles accumulate on the canvas creating a persistent painting effect
- Support multiple people simultaneously, each with their own rainbow color phase
- Press **ESC** to quit

**Note**: Move your hands in front of the Kinect sensor to pour paint onto the canvas. Faster movements create more particles with higher velocity. Particles follow physics (gravity, friction) and stick to the canvas when they stop moving, creating a lasting painting effect. The canvas slowly fades back to white over time, but stuck particles remain visible.

### Ring Collector Game

Run the ring collector game:
```bash
python kinect_v2_ring_collector.py
```

The application will:
- Initialize the Kinect v2 sensor (color video and body tracking)
- Display live color video feed from the Kinect camera as the background
- Overlay 15 gold rings randomly across the playable area (avoiding outer 10% on left/right sides)
- Track your body as a dark shadow skeleton overlaid on the video
- Detect when your hands "touch" rings to collect them
- Display score and timer in the top corners
- When all rings are collected:
  - Stop the timer
  - Display magnified score and completion time for 5 seconds
  - Show countdown from 10 to start the next round
  - Automatically reset and start a new game
- Press **ESC** to quit

**Game Features**:
- **Mixed Reality**: Gold rings overlaid on live Kinect color video feed - see yourself and your environment in real-time
- **Gold Rings**: Animated rings with pulsing and sparkle effects that appear to float in your space
- **Body Shadow**: Real-time skeleton shadow tracking your movements overlaid on the video
- **Collision Detection**: Rings disappear when your hand gets within touch distance
- **Timer**: Tracks your completion time (MM:SS.CS format)
- **Score Display**: Shows current score (collected/total rings)
- **Auto-Reset**: Game automatically resets after countdown for continuous play
- **Smart Placement**: Rings are spaced to avoid overlap and placed away from screen edges for better accessibility

**Note**: Move your hands in front of the Kinect sensor to collect the gold rings. You'll see yourself and your environment on screen with rings overlaid in mixed reality. Your body shadow will follow your movements in real-time. Touch rings with either hand to collect them. The game tracks your completion time and automatically starts a new round after you complete all rings.

### Hand on Ring Game

Run the hand on ring challenge game:
```bash
python kinect_v2_hand_on_ring.py
```

The application will:
- Initialize the Kinect v2 sensor (color video and body tracking)
- Display live color video feed from the Kinect camera as the background
- Start with a 10-second countdown instructing players to put their hands on colored rings
- Assign each tracked player a unique colored ring
- When countdown ends:
  - Rings with hands on them start moving at increasing speeds
  - Rings without hands disappear
  - Players must keep their hands on their rings as they move
- In multi-player mode:
  - Last player with hand on ring wins
  - Players eliminated if hand leaves ring for 2+ seconds
- In single-player mode (if only one player has hand on ring):
  - 20-second challenge mode activates
  - Ring moves at increasing speed
  - Player must maintain contact for full 20 seconds to win
  - Any break in contact results in loss
- Winner celebration displays for 10 seconds
- 10-second countdown before next round
- Automatically resets and starts new game
- Press **ESC** to quit

**Game Features**:
- **Mixed Reality**: Colored rings overlaid on live Kinect color video feed
- **Multi-Player Mode**: Competitive elimination with moving rings
- **Single-Player Challenge**: 20-second endurance challenge with accelerating ring movement
- **Progressive Difficulty**: Ring speed increases over time in both modes (faster in single-player)
- **Body Shadow**: Real-time skeleton shadow tracking overlaid on video
- **Hand Tracking**: Precise detection of hand position relative to rings
- **Elimination System**: Players eliminated if hand leaves ring for 2+ seconds (multi-player) or any break (single-player)
- **Smart Ring Movement**: Rings bounce off walls and change direction randomly
- **Color-Coded Players**: Each player gets a unique color (Blue, Green, Red, Yellow, Magenta, Cyan)
- **Auto-Reset**: Game automatically resets after winner celebration for continuous play

**Note**: Stand in front of the Kinect sensor during the 10-second countdown and place your hand on your assigned colored ring. Once the game starts, keep your hand on your ring as it moves! The ring will move faster over time, making it increasingly challenging. In single-player mode, you must maintain contact for the full 20 seconds to win the challenge. Your body shadow will track your movements in real-time on the video feed.

### Ice Hockey Game

Run the ice hockey game:
```bash
python kinect_v2_hockey.py
```

The application will:
- Initialize the Kinect v2 sensor (color video, depth, and body tracking)
- Display a top-down view of a hockey rink with transparent ice overlay
- Show live Kinect camera feed visible through the transparent rink
- Track your body position in 3D space and map it to the rink as a paddle
- Control a puck that moves with realistic physics (low friction for ice-like sliding)
- Detect collisions between your paddle (body) and the puck
- Score goals when the puck enters either goal area
- Display score (left vs right goals) at the top center
- Press **ESC** to quit

**Game Features**:
- **Top-Down View**: Bird's-eye view of the hockey rink using 3D body tracking
- **Mixed Reality**: Transparent ice rink overlay (40% opacity) showing live Kinect camera feed behind it
- **Body Tracking**: Your body acts as a paddle - move left/right and forward/backward to control it
- **Physics-Based Puck**: Realistic puck movement with low friction (0.995) for ice-like sliding
- **Collision Detection**: Elastic collisions between paddle and puck with velocity transfer
- **Goal Scoring**: Score points when puck enters left or right goal areas
- **Visual Feedback**: Paddle shows movement direction indicator, puck has highlight effect
- **3D Position Mapping**: Uses Kinect 3D joint positions (feet/spine) for accurate top-down mapping
- **Rink Design**: Includes center line, center circle, goal areas, and boundary walls

**Note**: Stand in front of the Kinect sensor and move your body to control the paddle. The game uses your feet position (or spine base) to determine your location on the rink. Move left/right to move the paddle horizontally, and move forward/backward to move it vertically on the rink. Hit the puck to send it flying! The puck slides with low friction like real ice hockey. Score by getting the puck into either goal area. The transparent rink allows you to see your environment while playing.

## Dependencies

- `pykinect2`: Python wrapper for Kinect v2 SDK
- `opencv-python`: Computer vision library for image processing and display
- `numpy`: Numerical computing library
- `comtypes`: COM type support for Windows

## Project Structure

```
kinectApps/
├── README.md                    # This file
├── environment.yml              # Conda environment configuration
├── kinect_v2_full_test.py      # Full test application with color, depth, and skeleton
├── kinect_stick_figure.py      # Stick figure animation controlled by Kinect movement
├── kinect_v2_paint_canvas.py   # Interactive paint canvas with particle physics
├── kinect_v2_ring_collector.py # Ring collector game with mixed reality and body tracking
├── kinect_v2_hand_on_ring.py   # Competitive hand-on-ring challenge game
└── kinect_v2_hockey.py         # Top-down ice hockey game with transparent rink overlay
```

## Notes

- Ensure the Kinect v2 sensor is properly connected and powered before running the application
- The applications track up to 6 bodies simultaneously (Kinect v2 limit)
- **kinect_v2_full_test.py**: 
  - Joints are displayed as circles: green for tracked, yellow for inferred
  - Bones are drawn as green lines connecting tracked joints
- **kinect_stick_figure.py**:
  - Draws clean stick figures on a black background
  - Tracks up to 6 people simultaneously (Kinect v2 limit)
  - Each person gets a distinct color for easy identification
  - Automatically scales and centers all figures for optimal viewing
  - Uses per-person smoothing to create fluid animation
  - Calculates combined bounding box to fit all figures in view
- **kinect_v2_paint_canvas.py**:
  - Creates a white canvas for painting
  - Uses hand positions (or body center if hands not tracked) as paint sources
  - Generates paint particles based on movement velocity
  - Particles are larger (10-12 pixels) for better visibility
  - Particles stick to canvas when they stop moving
  - Rainbow colors cycle continuously, with each person having a different color phase
  - Particles follow physics: gravity pulls them down, friction slows them
  - Canvas slowly fades back to white over time, but stuck particles persist
  - Supports up to 6 people simultaneously (Kinect v2 limit)
- **kinect_v2_ring_collector.py**:
  - Mixed reality game overlaying rings on live Kinect color video feed
  - 15 gold rings placed randomly with smart spacing algorithm
  - Rings avoid outer 10% of left/right screen edges for better accessibility
  - Real-time body shadow tracking (skeleton overlay) on video background
  - Hand-based collision detection for ring collection
  - Timer tracks completion time with centisecond precision
  - Score display shows collected/total rings
  - Magnified results screen after completion (5 seconds)
  - Countdown from 10 before next round
  - Automatic game reset for continuous play
  - Rings feature pulsing animation and rotating sparkle effects
  - Supports multiple people simultaneously (each person can collect rings)
  - Uses Kinect color frame (1920x1080) as canvas for true mixed reality experience
- **kinect_v2_hand_on_ring.py**:
  - Competitive elimination game with mixed reality overlay
  - 10-second start countdown with instructions
  - Each player assigned unique colored ring (up to 6 players)
  - Rings move at increasing speed over time
  - Multi-player mode: last player with hand on ring wins
  - Single-player challenge mode: 20-second endurance test
  - Progressive difficulty: speed increases faster in single-player mode (0.2x/sec vs 0.1x/sec)
  - Elimination: 2-second grace period in multi-player, instant in single-player
  - Rings bounce off walls and change direction randomly
  - Real-time body shadow tracking on video background
  - Winner celebration (10 seconds) followed by new round countdown
  - Automatic game reset for continuous play
  - Uses Kinect color frame (1920x1080) as canvas for mixed reality
  - Supports up to 6 people simultaneously (Kinect v2 limit)
- **kinect_v2_hockey.py**:
  - Top-down ice hockey game with mixed reality transparent rink overlay
  - Uses 3D body tracking (feet/spine positions) for accurate top-down mapping
  - Body acts as paddle - move left/right and forward/backward to control
  - Physics-based puck with low friction (0.995) for realistic ice-like sliding
  - Elastic collision detection between paddle and puck with velocity transfer
  - Goal scoring system with cooldown to prevent rapid scoring
  - Transparent rink overlay (40% opacity) showing live Kinect camera feed
  - Rink features: center line, center circle, goal areas, boundary walls
  - Score display (left vs right goals) at top center
  - Puck resets to center after each goal with random velocity
  - Paddle shows movement direction indicator for visual feedback
  - Tracks first detected body as player paddle
  - Uses Kinect color, depth, and body tracking simultaneously

## Troubleshooting

If you encounter initialization errors:
- Verify the Kinect v2 is connected via USB 3.0
- Ensure the Kinect v2 SDK is installed on your system
- Check that no other applications are using the Kinect sensor
