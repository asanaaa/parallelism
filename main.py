import threading
import time
import logging
import argparse
import os
import queue

import cv2
import numpy as np


LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sensor.log"),
    level=logging.ERROR,
    format="%(asctime)s : %(levelname)s : %(message)s"
)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    def __init__(self, delay: float, data_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._delay = delay
        self._data = 0
        self._stop_event = stop_event
        self.data_queue = data_queue
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            time.sleep(self._delay)
            self._data += 1
            if self.data_queue.full():
                self.data_queue.get()
            self.data_queue.put(self._data)

    def get(self):
        return self.data_queue.get_nowait() if not self.data_queue.empty() else self._data



class Camera:
    def __init__(self, cam_name, resolution, fps, stop_event: threading.Event):
        self.cam_name = cam_name
        self.width, self.height = map(int, resolution.split('x'))
        self.fps = fps
        self.cap = None
        self.last_frame = None
        self._stop_event = stop_event
        self.last_frame_time = time.time()
        self.frame_queue = queue.Queue(maxsize=2)
        self.open_camera()
        self.thread = threading.Thread(target=self._base_loop, daemon=True)
        self.thread.start()

    def open_camera(self):
        if self.cap:
            self.cap.release()

        try:
            self.cap = cv2.VideoCapture(self.cam_name, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                stop_event.set()
                raise RuntimeError("Camera not found or disconnected!")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception as e:
            logging.error(f"Error occurred: {e}")

    def _base_loop(self):
        while not self._stop_event.is_set():
            try:
                if not self.cap.isOpened():
                    stop_event.set()
                    raise RuntimeError("Camera not found or disconnected!")
                
                ret, frame = self.cap.read()
                if not ret:
                    stop_event.set()
                    raise RuntimeError("Failed to read frame.")

                self.last_frame = frame

                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)

                elapsed = time.time() - self.last_frame_time
                delay = max(0, (1.0 / self.fps) - elapsed)
                time.sleep(delay)
                self.last_frame_time = time.time()

            except Exception as e:
                logging.error(f"Exception occurred while reading frame: {e}")

    def get_frame(self):
        return self.frame_queue.get_nowait() if not self.frame_queue.empty() else self.last_frame

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

class SensorCam(Sensor):
    def __init__(self, camera: Camera):
        self.camera = camera

    def get(self):
        return self.camera.get_frame()

class WindowImage:
    def show(self, img):
        if img is not None:
            cv2.imshow("Sensor Data", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    def put_data(self, frame, sensor_data):
        new_frame = frame.copy()
        height, width, _ = frame.shape
        y_offset = height - 60
        x_offset = width - 250

        for idx, data in enumerate(sensor_data):
            text = f"Sensor {idx + 1}: {data}"
            cv2.putText(new_frame, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            y_offset -= 30
        return new_frame
    
    def __del__(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, required=True, help="Camera name.")
    parser.add_argument("--res", type=str, required=True, help="Resolution.")
    parser.add_argument("--fps", type=int, required=True, help="Frame per second.")
    args = parser.parse_args()

    try:
        stop_event = threading.Event()
        camera = Camera(args.cam, args.res, args.fps, stop_event)
        cam_sensor = SensorCam(camera)
        sensor_queues = [queue.Queue(maxsize=2) for _ in range(3)]

        sensors = [
            SensorX(0.01, sensor_queues[0], stop_event),
            SensorX(0.1, sensor_queues[1], stop_event),
            SensorX(1.0, sensor_queues[2], stop_event)
        ]
        window = WindowImage()

        while not stop_event.is_set():
            frame = cam_sensor.get()
            if frame is None:
                frame = np.ones((camera.height, camera.width, 3), dtype=np.uint8) * 255

            sensor_data = [sensor.get() for sensor in sensors]
            # print("Sensor Data:", sensor_data)

            frame_with_data = window.put_data(frame, sensor_data)

            if not window.show(frame_with_data):
                # print("break")
                stop_event.set()
                break
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
    finally:
        try:
            del camera
            del window
        except Exception as e:
            logging.error(f"Deletion error: {e}")

