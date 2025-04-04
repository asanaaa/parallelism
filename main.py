import cv2
import os
import time
import argparse
import logging
from threading import Thread
from ultralytics import YOLO
from multiprocessing import Process, Manager


logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

def process_batch(index, frame_chunk, processed_batches):
    model = YOLO("yolov8m-pose.pt")
    result_chunk = []
    for frame in frame_chunk:
        results = model(frame)
        if results:
            processed_frame = frame.copy()
            for result in results:
                processed_frame = result.plot()
            result_chunk.append(processed_frame)
        else:
            result_chunk.append(frame)
    processed_batches[index] = result_chunk

class Reader:
    def __init__(self, video_path, output_name, num_threads, batch_size):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File {video_path} not found.")

        self.video_path = video_path
        self.output_name = output_name
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.frames = []
        
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError("Can't open video.")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_name, self._fourcc, fps, (width, height))

    def read(self):
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                print("The end of the video reached or cannot read frame.")
                break
            self.frames.append(frame.copy())

    def run_single(self):
        model = YOLO("yolov8m-pose.pt")
        processed_frames = []
        start_time = time.time()
        for frame in self.frames:
            results = model(frame)
            if results:
                processed_frame = frame.copy()
                for result in results:
                    processed_frame = result.plot()
                processed_frames.append(processed_frame)
            else:
                processed_frames.append(frame)

        full_time = time.time() - start_time
        for frame in processed_frames:
            self.out.write(frame)

        print(f"Processing for 1 thread finished in {full_time:.2f} seconds.")
        

    def run(self):
        if not self.frames:
            print("No frames to process.")
            return

        start_time = time.time()
        new_frames = []

        with Manager() as manager:
            processed_batches = manager.list([None] * self.num_threads)

            total_frames = len(self.frames)
            chunk_size = total_frames // self.num_threads
            frame_chunks = [self.frames[i * chunk_size : (i + 1) * chunk_size] for i in range(self.num_threads - 1)]
            frame_chunks.append(self.frames[(self.num_threads - 1) * chunk_size :])

            processes = []
            for i in range(self.num_threads):
                proc = Process(target=process_batch, args=(i, frame_chunks[i], processed_batches))
                proc.start()
                processes.append(proc)

            for proc in processes:
                proc.join()

            full_time = time.time() - start_time

            for batch in processed_batches:
                new_frames.extend(batch)

            for frame in new_frames:
                self.out.write(frame)

            print(f"Processing for {self.num_threads} processes finished in {full_time:.2f} seconds.")

    def __del__(self):
        self._cap.release()
        self.out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to input video.")
    parser.add_argument("--threads", type=int, required=True, help="Number of threads.")
    parser.add_argument("--name", type=str, required=True, help="Name of output file.")
    args = parser.parse_args()

    reader = Reader(args.path, args.name, args.threads, 30)
    reader.read()
    # if args.threads == 1:
    #     reader.run_single()
    # else:
    #     reader.run()
    reader.run()