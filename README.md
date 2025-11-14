# Kinect v2 Applications

A Python application for capturing and visualizing data from Microsoft Kinect v2, including color frames, depth maps, and body skeleton tracking.

## Features

- **Color Frame Capture**: Real-time color video feed from Kinect v2
- **Depth Frame Visualization**: Depth map visualization with color mapping
- **Body Skeleton Tracking**: Real-time body pose detection and skeleton overlay on color frames
- **Multi-frame Support**: Simultaneous capture of color, depth, and body tracking data

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
└── kinect_v2_full_test.py      # Main application script
```

## Notes

- Ensure the Kinect v2 sensor is properly connected and powered before running the application
- The application tracks up to 6 bodies simultaneously (Kinect v2 limit)
- Joints are displayed as circles: green for tracked, yellow for inferred
- Bones are drawn as green lines connecting tracked joints

## Troubleshooting

If you encounter initialization errors:
- Verify the Kinect v2 is connected via USB 3.0
- Ensure the Kinect v2 SDK is installed on your system
- Check that no other applications are using the Kinect sensor
