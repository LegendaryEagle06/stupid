import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time


# Load the YOLO model
def load_model():
    net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use GPU if available: DNN_TARGET_CUDA
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes


# Threaded video capture class for better performance
class ThreadedCamera:
    def __init__(self, cam_index):
        self.cam_index = cam_index
        self.cap = None
        self.ret, self.frame = False, None
        self.stopped = False
        self.reconnecting = False
        self.initialize_camera()
        threading.Thread(target=self.update, args=(), daemon=True).start()

    def initialize_camera(self):
        """Initialize or reinitialize the camera."""
        print(f"[DEBUG] Initializing camera {self.cam_index}...")
        if self.cap is not None:
            self.cap.release()
            print(f"[DEBUG] Released existing camera resource for {self.cam_index}.")
            time.sleep(0.5)  # Wait for resources to be freed

        self.cap = cv2.VideoCapture(self.cam_index)
        retries = 5  # Retry up to 5 times
        for attempt in range(retries):
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if self.ret:
                    print(f"[DEBUG] Camera {self.cam_index} initialized successfully on attempt {attempt + 1}.")
                    return
            print(f"[DEBUG] Retrying camera {self.cam_index} ({attempt + 1}/{retries})...")
            time.sleep(1)  # Allow the camera to initialize
        else:
            print(f"[ERROR] Failed to initialize Camera {self.cam_index} after {retries} attempts.")

    def update(self):
        """Continuously read frames from the camera."""
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                self.ret = False
                if not self.reconnecting:
                    threading.Thread(target=self.reconnect, args=(), daemon=True).start()

    def reconnect(self):
        """Reconnect the camera asynchronously."""
        if self.stopped:
            return  # Skip reconnect if the camera is explicitly stopped
        self.reconnecting = True
        print(f"[DEBUG] Attempting to reconnect camera {self.cam_index}...")
        self.initialize_camera()
        self.reconnecting = False

    def read(self):
        """Return the latest frame."""
        return self.ret, self.frame

    def stop(self):
        """Stop the camera feed."""
        self.stopped = True
        if self.cap:
            self.cap.release()
            print(f"[DEBUG] Camera {self.cam_index} stopped and resources released.")


# Detect available cameras
def detect_cameras(max_index=10):
    available_cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
        cap.release()
    return available_cameras


# Unified GUI for video feed and controls
class CameraControlApp:
    def __init__(self, root, available_cameras):
        self.root = root
        self.root.title("Multi-Camera Tracking")
        self.root.geometry("1280x800")

        # Initialize YOLO model
        self.net, self.classes = load_model()

        # Initialize attributes
        self.available_cameras = available_cameras
        self.active_cameras = {}
        self.frame_counts = {}
        self.detection_interval = 5
        self.frames = []
        self.canvas = None
        self.quit_flag = False
        self.camera_lock = threading.Lock()

        # Create main layout
        self.create_layout()

        # Start video update loop
        self.update_video_feed()

    def create_layout(self):
        # Video display area
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas for video feed
        self.canvas = tk.Canvas(self.video_frame, width=1280, height=720, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control bar
        self.control_frame = tk.Frame(self.root, height=80, bg="white")
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create camera toggles
        self.camera_vars = {}
        for cam_index in self.available_cameras:
            var = tk.BooleanVar(value=True)
            self.camera_vars[cam_index] = var
            checkbox = tk.Checkbutton(
                self.control_frame,
                text=f"Camera {cam_index}",
                variable=var,
                command=lambda i=cam_index: self.toggle_camera(i, var.get()),
            )
            checkbox.pack(side=tk.LEFT, padx=10, pady=10)

        # Quit button
        quit_button = tk.Button(self.control_frame, text="Quit", command=self.quit_program)
        quit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initialize selected cameras
        for cam_index in self.available_cameras:
            self.toggle_camera(cam_index, True)

    def toggle_camera(self, cam_index, enable):
        with self.camera_lock:
            # Debugging: Log the current state and requested action
            current_state = "Enabled" if cam_index in self.active_cameras else "Disabled"
            requested_action = "Enable" if enable else "Disable"
            print(
                f"[DEBUG] Toggling Camera {cam_index} - Current State: {current_state}, Requested: {requested_action}")

            # Avoid redundant actions
            if enable and cam_index in self.active_cameras and not self.active_cameras[cam_index].stopped:
                print(f"[DEBUG] No action taken. Camera {cam_index} is already enabled and active.")
                return
            if not enable and cam_index not in self.active_cameras:
                print(f"[DEBUG] No action taken. Camera {cam_index} is already disabled.")
                return

            if enable:
                # Enable the camera
                try:
                    print(f"[DEBUG] Enabling Camera {cam_index}...")
                    if cam_index in self.active_cameras:
                        # Stop and remove the existing instance if any
                        self.active_cameras[cam_index].stop()
                        del self.active_cameras[cam_index]

                    # Create a new ThreadedCamera instance
                    self.active_cameras[cam_index] = ThreadedCamera(cam_index)
                    self.frame_counts[cam_index] = 0
                    print(f"[DEBUG] Camera {cam_index} successfully enabled and initialized.")
                except Exception as e:
                    print(f"[ERROR] Failed to enable Camera {cam_index}: {e}")
            else:
                # Disable the camera
                try:
                    print(f"[DEBUG] Disabling Camera {cam_index}...")
                    if cam_index in self.active_cameras:
                        self.active_cameras[cam_index].stop()
                        del self.active_cameras[cam_index]
                        del self.frame_counts[cam_index]
                        print(f"[DEBUG] Camera {cam_index} successfully disabled.")
                    else:
                        print(f"[DEBUG] Camera {cam_index} was not active.")
                except Exception as e:
                    print(f"[ERROR] Failed to disable Camera {cam_index}: {e}")

            # Log post-toggle state
            post_toggle_state = "Enabled" if cam_index in self.active_cameras else "Disabled"
            print(f"[DEBUG] Post-Toggle State - Camera {cam_index}: {post_toggle_state}")

    def quit_program(self):
        self.quit_flag = True
        for camera in self.active_cameras.values():
            camera.stop()
        self.root.destroy()

    def update_video_feed(self):
        if self.quit_flag:
            return

        # Process frames from active cameras
        frames = []
        for cam_index, camera in self.active_cameras.items():
            ret, frame = camera.read()
            if ret:
                self.frame_counts[cam_index] += 1

                # Run YOLO detection every few frames
                if self.frame_counts[cam_index] % self.detection_interval == 0:
                    frame = self.run_yolo_detection(frame)

                frames.append(frame)

        # Display frames on canvas
        if frames:
            grid_frame = self.create_dynamic_grid(frames, 1280, 720)
            img = Image.fromarray(cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Prevent garbage collection

        # Schedule the next update
        self.root.after(10, self.update_video_feed)

    def run_yolo_detection(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        # Extract boxes, confidences, and class IDs
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def create_dynamic_grid(self, frames, window_width, window_height):
        num_frames = len(frames)
        grid = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        if num_frames == 0:
            return grid

        if num_frames == 1:
            # Fullscreen
            grid = cv2.resize(frames[0], (window_width, window_height))
        elif num_frames == 2:
            # Side by side
            half_width = window_width // 2
            resized_frames = [cv2.resize(f, (half_width, window_height)) for f in frames]
            grid[:, :half_width] = resized_frames[0]
            grid[:, half_width:] = resized_frames[1]
        elif num_frames == 3:
            # Two on top, one on bottom
            half_width = window_width // 2
            half_height = window_height // 2
            resized_frames = [cv2.resize(f, (half_width, half_height)) for f in frames[:2]]
            grid[:half_height, :half_width] = resized_frames[0]
            grid[:half_height, half_width:] = resized_frames[1]
            grid[half_height:, :] = cv2.resize(frames[2], (window_width, half_height))
        elif num_frames >= 4:
            # 2x2 grid
            half_width = window_width // 2
            half_height = window_height // 2
            resized_frames = [cv2.resize(f, (half_width, half_height)) for f in frames[:4]]
            grid[:half_height, :half_width] = resized_frames[0]
            grid[:half_height, half_width:] = resized_frames[1]
            grid[half_height:, :half_width] = resized_frames[2]
            grid[half_height:, half_width:] = resized_frames[3]

        return grid


# Main function
if __name__ == "__main__":
    available_cameras = detect_cameras(max_index=10)
    if not available_cameras:
        print("No cameras detected. Exiting...")
    else:
        root = tk.Tk()
        app = CameraControlApp(root, available_cameras)
        root.mainloop()
