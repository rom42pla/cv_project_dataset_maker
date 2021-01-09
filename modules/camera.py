from itertools import product

from threading import Thread
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class Camera:

    def __init__(self,
                 cam_number: int = 0, n_frames: int = 30, resolution: int = 720,
                 show_fps: bool = True,
                 window_size: int = 175,
                 window_recording_color: tuple = (255, 0, 0), window_not_recording_color: tuple = (0, 255, 0)):
        # sets camera's properties
        self.is_running = False
        self.vid, self.thread = cv2.VideoCapture(cam_number), None

        # sets the resolution of the webcam
        assert isinstance(resolution, int) or isinstance(resolution, tuple) or isinstance(resolution, list)
        if isinstance(resolution, int):
            resolution = (int((resolution * 16) / 9), resolution)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        _, initial_frame = self.vid.read()
        self.resolution = initial_frame.shape[1], initial_frame.shape[0]

        # whether or not to show FPS label
        assert isinstance(show_fps, bool)
        self.show_fps = show_fps

        # settings about the recording square window
        assert isinstance(window_recording_color, tuple) or isinstance(window_recording_color, list)
        assert [_ for color in window_recording_color if color < 256] is not []
        assert [_ for color in window_not_recording_color if color < 256] is not []
        assert window_size < self.resolution[0] // 2
        self.window_recording_color, self.window_not_recording_color = window_recording_color, \
                                                                       window_not_recording_color
        self.window_center = (self.resolution[0] // 2, self.resolution[1] // 2)

        # structures used to save the various frames
        self.n_frames, self.window_size = n_frames, window_size
        self.q_frames = np.zeros(shape=(n_frames, self.window_size * 2, self.window_size * 2, 3))

    def frame_elaboration(self, save_frame):
        show_frame = np.copy(save_frame)

        # horizontally flips the image
        show_frame = np.flip(show_frame, axis=1)

        # draws a rectangle on the app
        show_frame = cv2.rectangle(np.array(show_frame),
                                   (self.window_center[0] - self.window_size, self.window_center[1] - self.window_size),
                                   (self.window_center[0] + self.window_size, self.window_center[1] + self.window_size),
                                   color=self.window_not_recording_color, thickness=2)
        # crops the area in the rectangle
        save_frame = save_frame[self.window_center[1] - self.window_size: self.window_center[1] + self.window_size,
                     self.window_center[0] - self.window_size: self.window_center[0] + self.window_size]

        show_frame = np.array(show_frame)

        return show_frame, save_frame

    def capture_frame(self):
        prev_frame_time, new_frame_time = 0, 0

        while self.is_running:
            try:
                ret, frame = self.vid.read()
                show_frame, save_frame = self.frame_elaboration(frame)
            except Exception as exception:
                print(exception)
                break

            # saves the frame in memory
            # save_frame = cv2.cvtColor(save_frame, cv2.COLOR_BGR2GRAY)
            self.q_frames[:-1] = self.q_frames[1:]
            self.q_frames[-1] = save_frame

            # eventually shows FPS on the app
            if self.show_fps:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(img=show_frame, text=f"FPS: {np.round(fps)}",
                            org=(0, 30), color=(0, 255, 0),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
            cv2.imshow('ASL Dataset Maker', show_frame)

            if cv2.waitKey(1) & False:
                pass

    def start(self):
        if self.is_running:
            raise Exception(f"Camera is already open")
        # starts the recording thread
        self.is_running = True
        self.thread = Thread(target=self.capture_frame)
        self.thread.start()

    def stop(self):
        # stops the recording thread
        self.is_running = False
        self.vid.release()
        cv2.destroyAllWindows()

    def plot_last_frames(self):
        def get_optimal_dimensions(samples):
            possible_combinations = []
            # tries every possible combination of numbers
            for n1, n2 in product(range(1, int(np.ceil(np.sqrt(samples))) * 2),
                                  range(1, int(np.ceil(np.sqrt(samples))) * 2)):
                # return the first matching combination
                if n1 * n2 == samples:
                    possible_combinations += [(n1, n2)]
            if possible_combinations is not []:
                distances = [np.abs(n1 - n2) for n1, n2 in possible_combinations]
                n1, n2 = possible_combinations[distances.index(min(distances))]
                return np.asarray([np.max([n1, n2]), np.min([n1, n2])], dtype=np.int)
            # else return a sure resolution
            return np.asarray([int(np.ceil(np.sqrt(samples))),
                               int(np.ceil(np.sqrt(samples)))], dtype=np.int)

        # sets image's parameters
        n_samples, img_shape = int(self.q_frames.shape[0]), (self.q_frames.shape[1], self.q_frames.shape[2])
        figsize = get_optimal_dimensions(n_samples)
        fig, axs = plt.subplots(*figsize, sharex="all", sharey="all", gridspec_kw={'hspace': 0, 'wspace': 0})
        for i_ax, ax in enumerate(axs.flat):
            # removes the labels from the image
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            # checks if the limit is reached
            if i_ax >= self.q_frames.shape[0]:
                continue
            # fill the axis with the image
            ax.imshow(self.q_frames[i_ax])  # , cmap="gray")
        # shows the plot
        plt.show()
