import pyrealsense2 as rs
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LINE_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)

# --- Global Variables ---
selected_points_2d = [] # Stores 2D points selected by mouse click for current measurement
current_measurement_3d_points = [] # Stores corresponding 3D points for current measurement
measurements_log = [] # Log of all completed measurements: [{'label': str, 'points_3d': [p1, p2], 'distance': float}]
intrinsics = None # Camera intrinsics
depth_scale = 1.0 # Depth scale for the depth sensor
current_label_input = "" # For typing label for measurement
typing_mode = False # True when user is typing a label

# --- Helper Functions ---
def deproject_pixel_to_point_custom(pixel, depth, intrinsics_obj):
    """
    Deprojects a 2D pixel with its depth value to a 3D point.
    :param pixel: [x, y] pixel coordinates
    :param depth: depth value at that pixel (in meters)
    :param intrinsics_obj: rs.intrinsics object
    :return: [X, Y, Z] 3D point in camera coordinates (in meters)
    """
    x = (pixel[0] - intrinsics_obj.ppx) / intrinsics_obj.fx
    y = (pixel[1] - intrinsics_obj.ppy) / intrinsics_obj.fy
    return [depth * x, depth * y, depth]

def calculate_3d_distance(p1, p2):
    """Calculates Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks to select points."""
    global selected_points_2d, current_measurement_3d_points, intrinsics, depth_frame_global, typing_mode

    if typing_mode: # Don't process clicks if typing label
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if depth_frame_global and intrinsics:
            if len(selected_points_2d) < 2:
                depth_value = depth_frame_global.get_distance(x, y)
                if depth_value == 0:
                    print(f"Warning: Depth is 0 at selected point ({x}, {y}). Point not added. Try a different point.")
                    return

                selected_points_2d.append((x, y))
                # point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value) # SDK's deprojection
                point_3d = deproject_pixel_to_point_custom([x,y], depth_value, intrinsics) # Custom for clarity
                current_measurement_3d_points.append(point_3d)
                print(f"Point {len(selected_points_2d)} selected: 2D=({x},{y}), 3D=({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m")

                if len(selected_points_2d) == 2:
                    dist = calculate_3d_distance(current_measurement_3d_points[0], current_measurement_3d_points[1])
                    print(f"Distance between points: {dist:.3f} meters")
                    print("Press 's' to save this measurement with a label, or 'c' to clear points.")
            else:
                print("Already 2 points selected. Press 's' to save or 'c' to clear.")
        else:
            print("Depth frame or intrinsics not available yet.")

def draw_ui(image):
    """Draws UI elements and measurement information on the image."""
    global selected_points_2d, current_measurement_3d_points, measurements_log, typing_mode, current_label_input

    y_offset = 30
    # Display instructions
    instructions = [
        "Click 2 points to measure distance.",
        "Press 's' to save measurement (after 2 points).",
        "Press 'c' to clear current points.",
        "Press 'x' to clear ALL saved measurements.",
        "Press 'p' to generate 3-view plot of saved measurements.",
        "Press 'q' to quit."
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(image, instruction, (10, y_offset + i * 25), FONT, 0.6, TEXT_BG_COLOR, FONT_THICKNESS + 2)
        cv2.putText(image, instruction, (10, y_offset + i * 25), FONT, 0.6, TEXT_COLOR, FONT_THICKNESS)


    # Draw current selection
    for i, p_2d in enumerate(selected_points_2d):
        cv2.circle(image, p_2d, 5, POINT_COLOR, -1)
        cv2.putText(image, f"P{i+1}", (p_2d[0] + 10, p_2d[1] - 10), FONT, FONT_SCALE, POINT_COLOR, FONT_THICKNESS)
        if len(current_measurement_3d_points) > i:
            p_3d_text = f"({current_measurement_3d_points[i][0]:.2f}, {current_measurement_3d_points[i][1]:.2f}, {current_measurement_3d_points[i][2]:.2f})m"
            cv2.putText(image, p_3d_text, (p_2d[0] + 10, p_2d[1] + 15), FONT, 0.5, POINT_COLOR, 1)


    if len(selected_points_2d) == 2:
        cv2.line(image, selected_points_2d[0], selected_points_2d[1], LINE_COLOR, 2)
        dist = calculate_3d_distance(current_measurement_3d_points[0], current_measurement_3d_points[1])
        mid_point = ((selected_points_2d[0][0] + selected_points_2d[1][0]) // 2,
                       (selected_points_2d[0][1] + selected_points_2d[1][1]) // 2)
        cv2.putText(image, f"{dist:.3f} m", (mid_point[0], mid_point[1] - 10), FONT, FONT_SCALE, LINE_COLOR, FONT_THICKNESS)

    # Display logged measurements
    log_y_start = y_offset + len(instructions) * 25 + 10
    cv2.putText(image, "Logged Measurements:", (10, log_y_start), FONT, 0.6, TEXT_BG_COLOR, FONT_THICKNESS +2)
    cv2.putText(image, "Logged Measurements:", (10, log_y_start), FONT, 0.6, TEXT_COLOR, FONT_THICKNESS)
    for i, log in enumerate(measurements_log):
        text = f"- {log['label']}: {log['distance']:.3f} m"
        cv2.putText(image, text, (10, log_y_start + 20 + i * 25), FONT, 0.6, TEXT_BG_COLOR, FONT_THICKNESS+2)
        cv2.putText(image, text, (10, log_y_start + 20 + i * 25), FONT, 0.6, TEXT_COLOR, FONT_THICKNESS)

    # Display typing mode for label
    if typing_mode:
        cv2.putText(image, f"Enter label: {current_label_input}", (10, image.shape[0] - 30), FONT, FONT_SCALE, TEXT_BG_COLOR, FONT_THICKNESS+2)
        cv2.putText(image, f"Enter label: {current_label_input}", (10, image.shape[0] - 30), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

def plot_three_view(measurements):
    if not measurements:
        print("No measurements to plot.")
        return

    fig = plt.figure(figsize=(15, 5))
    ax_front = fig.add_subplot(131, aspect='equal') # YZ plane (Front view)
    ax_top = fig.add_subplot(132, aspect='equal')   # XZ plane (Top view)
    ax_side = fig.add_subplot(133, aspect='equal')  # XY plane (Side view)

    ax_front.set_xlabel("Y (m)")
    ax_front.set_ylabel("Z (m)")
    ax_front.set_title("Front View (YZ plane)")

    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Z (m)")
    ax_top.set_title("Top View (XZ plane)")

    ax_side.set_xlabel("X (m)")
    ax_side.set_ylabel("Y (m)")
    ax_side.set_title("Side View (XY plane)")

    all_x, all_y, all_z = [], [], []

    for meas in measurements:
        p1 = meas['points_3d'][0]
        p2 = meas['points_3d'][1]
        label = meas['label']

        all_x.extend([p1[0], p2[0]])
        all_y.extend([p1[1], p2[1]])
        all_z.extend([p1[2], p2[2]])

        # Front View (Y, Z)
        ax_front.plot([p1[1], p2[1]], [p1[2], p2[2]], marker='o', label=label)
        # Top View (X, Z)
        ax_top.plot([p1[0], p2[0]], [p1[2], p2[2]], marker='o', label=label)
        # Side View (X, Y)
        ax_side.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o', label=label)

    # Auto-scaling for axes based on all points
    if all_x: # Check if any points were plotted
        # To make scales comparable and centered around data
        def set_axes_limits(ax, data1, data2):
            if not data1 or not data2: return
            min1, max1 = min(data1), max(data1)
            min2, max2 = min(data2), max(data2)
            range1, range2 = max1 - min1, max2 - min2
            max_range = max(range1, range2, 0.1) # ensure some minimal range
            mid1, mid2 = (min1+max1)/2, (min2+max2)/2
            ax.set_xlim(mid1 - max_range/2, mid1 + max_range/2)
            ax.set_ylim(mid2 - max_range/2, mid2 + max_range/2)

        set_axes_limits(ax_front, all_y, all_z)
        set_axes_limits(ax_top, all_x, all_z)
        set_axes_limits(ax_side, all_x, all_y)

    ax_front.legend()
    ax_top.legend()
    ax_side.legend()

    plt.tight_layout()
    plt.show()


# --- Main Function ---
depth_frame_global = None # To make it accessible in mouse_callback

def main():
    global selected_points_2d, current_measurement_3d_points, measurements_log, intrinsics, depth_frame_global
    global depth_scale, typing_mode, current_label_input

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    # device_product_line = str(device.get_info(rs.camera_info.product_line)) # Not strictly needed for D455 known res

    # Check if D455 and set appropriate resolution if needed (D455 supports various)
    # For simplicity, using a common resolution. You might want to adjust.
    # Example: 848x480 or 1280x720
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("D455 RGB sensor not found.")
        return

    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get intrinsics for the color stream (after alignment, depth pixels correspond to color pixels)
    # We will deproject using depth values from the aligned depth frame,
    # but using the intrinsics of the stream the depth is aligned TO (the color stream).
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    print(f"Color Intrinsics: {intrinsics}")


    cv2.namedWindow('ROV Measurement - D455', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('ROV Measurement - D455', mouse_callback)

    print("Camera initialized. Click two points on the image to measure distance.")
    print("Press 's' to save the current measurement after selecting two points.")
    print("Press 'c' to clear current selection. 'x' to clear all saved. 'p' for 3-view plot. 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_frame_global = aligned_depth_frame # Make it accessible to callback

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (for visualization purposes)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03/depth_scale), cv2.COLORMAP_JET) # Adjust alpha for better viz

            display_image = color_image.copy()
            draw_ui(display_image)

            images = np.hstack((display_image, depth_colormap))
            cv2.imshow('ROV Measurement - D455', images)
            key = cv2.waitKey(1) & 0xFF

            if typing_mode:
                if key == 13: # Enter key
                    if current_label_input and len(current_measurement_3d_points) == 2:
                        dist = calculate_3d_distance(current_measurement_3d_points[0], current_measurement_3d_points[1])
                        measurements_log.append({
                            'label': current_label_input,
                            'points_3d': list(current_measurement_3d_points), # Store a copy
                            'distance': dist
                        })
                        print(f"Measurement saved: {current_label_input} - {dist:.3f} m")
                    typing_mode = False
                    current_label_input = ""
                    selected_points_2d.clear()
                    current_measurement_3d_points.clear()
                elif key == 27: # Escape key
                    typing_mode = False
                    current_label_input = ""
                    print("Labeling cancelled.")
                elif key >= 32 and key <= 126: # Printable ASCII
                    current_label_input += chr(key)
                elif key == 8: # Backspace
                    current_label_input = current_label_input[:-1]
                continue # Continue to next frame waitKey if typing

            if key == ord('q'):
                break
            elif key == ord('c'):
                selected_points_2d.clear()
                current_measurement_3d_points.clear()
                print("Current points cleared.")
            elif key == ord('x'):
                measurements_log.clear()
                print("All saved measurements cleared.")
            elif key == ord('s'):
                if len(current_measurement_3d_points) == 2:
                    typing_mode = True
                    current_label_input = "" # Reset label input
                    print("Enter a label for this measurement (e.g., Length, Width, Ruler_30cm) and press Enter:")
                else:
                    print("Please select 2 points first to define a measurement.")
            elif key == ord('p'):
                if measurements_log:
                    plot_three_view(measurements_log)
                else:
                    print("No saved measurements to plot.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")

if __name__ == "__main__":
    # Store depth_frame globally for access in callback (simplification for this example)
    # A class-based approach would be cleaner for managing state.
    main()