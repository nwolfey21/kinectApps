# Kinect v2 Applications

A collection of Python applications for capturing and visualizing data from Microsoft Kinect v2, including color frames, depth maps, body skeleton tracking, stick figure animation, and interactive paint canvas.

## Features

- **Color Frame Capture**: Real-time color video feed from Kinect v2
- **Depth Frame Visualization**: Depth map visualization with color mapping
- **Body Skeleton Tracking**: Real-time body pose detection and skeleton overlay on color frames
- **Multi-frame Support**: Simultaneous capture of color, depth, and body tracking data
- **Stick Figure Animation**: Real-time stick figure drawing controlled by Kinect body movement with support for multiple people (up to 6)
- **Paint Canvas**: Interactive fluid paint simulation where hand movements control paint flow with rainbow colors on a white canvas

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
└── kinect_v2_paint_canvas.py   # Interactive paint canvas with particle physics
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

## Troubleshooting

If you encounter initialization errors:
- Verify the Kinect v2 is connected via USB 3.0
- Ensure the Kinect v2 SDK is installed on your system
- Check that no other applications are using the Kinect sensor
