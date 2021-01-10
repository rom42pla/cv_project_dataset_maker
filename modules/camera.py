from os import makedirs, listdir
from os.path import exists, join
import re

from itertools import product

from pprint import pprint, pformat

from threading import Thread
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class Camera:

    def __init__(self, assets_path: str,
                 cam_number: int = 0, resolution: int = 720, show_fps: bool = True,
                 n_frames: int = 30,
                 seconds_to_be_recorded: float = 2, seconds_to_be_idle: float = 6,
                 window_size: int = 175,
                 window_recording_color: tuple = (0, 0, 255), window_not_recording_color: tuple = (0, 255, 0)):
        # eventually creates output directory
        self.assets_path = assets_path
        self.samples_path, self.letters_examples = join(assets_path, "samples"), join(assets_path, "letters_examples")
        if not exists(self.samples_path):
            makedirs(self.samples_path)
        if not exists(self.letters_examples):
            makedirs(self.letters_examples)

        # sets camera's properties
        self.is_running = False
        self.vid, self.thread = cv2.VideoCapture(cam_number), None

        # states' variables
        self.mimed_letters = []
        self.state_starting_time, self.seconds_to_be_recorded, self.seconds_to_be_idle = None, \
                                                                                         seconds_to_be_recorded, \
                                                                                         seconds_to_be_idle

        # creates the states' graph
        self.states_graph = CameraStatesGraph()
        self.states_graph.add_state("recording", seconds=self.seconds_to_be_recorded)
        self.states_graph.add_state("idle", seconds=self.seconds_to_be_idle)
        self.states_graph.add_edge(edge_from="idle", edge_to="recording")
        self.states_graph.add_edge(edge_from="recording", edge_to="idle")
        self.states_graph.set_current_state("idle")

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

    def frame_elaboration(self, save_frame, horizontal_flip: bool = True):
        # horizontally flips the image
        if horizontal_flip:
            save_frame = np.flip(save_frame, axis=1)

        show_frame = np.copy(save_frame)

        # draws a rectangle on the app
        window_color = self.window_not_recording_color if self.states_graph.current_state == "idle" \
            else self.window_recording_color
        show_frame = cv2.rectangle(show_frame,
                                   (self.window_center[0] - self.window_size, self.window_center[1] - self.window_size),
                                   (self.window_center[0] + self.window_size, self.window_center[1] + self.window_size),
                                   color=window_color, thickness=2)

        # shows the new letter to mimic
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        if len(self.mimed_letters) == 0:
            # self.mimed_letters += [alphabet[np.random.randint(0, len(alphabet))]]
            self.mimed_letters += ["a"]
        elif self.states_graph.is_new_state and self.states_graph.current_state == "idle":
            next_letter_index = alphabet.index(self.mimed_letters[-1]) + 1
            if next_letter_index >= len(alphabet):
                next_letter_index = 0
            self.mimed_letters += [alphabet[next_letter_index]]
        cv2.putText(img=show_frame, text=f"Letter '{self.mimed_letters[-1]}'",
                    org=(self.window_center[0] - self.window_size, self.window_center[1] - self.window_size - 25),
                    color=window_color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, thickness=1)

        # show a progress bar
        percentage = (time.time() - self.state_starting_time) / \
                     self.states_graph.get_seconds(self.states_graph.current_state)
        show_frame = cv2.rectangle(show_frame,
                                   (self.window_center[0] - self.window_size, self.window_center[1] - self.window_size),
                                   (self.window_center[0] - self.window_size + int(self.window_size * percentage * 2),
                                    25 + self.window_size),
                                   color=window_color, thickness=-1)

        # shows an image to use as example for the next letter
        letter_img = [img_name for img_name in listdir(self.letters_examples) if
                      re.match(f"{self.mimed_letters[-1]}\..*", img_name)]
        letter_img = join(self.letters_examples, letter_img[0]) if len(letter_img) > 0 else None
        if letter_img:
            letter_img = cv2.imread(letter_img)
            letter_img_new_dimensions = [128, (letter_img.shape[0] * 128) // letter_img.shape[1]]
            if letter_img_new_dimensions[1] % 2 != 0: letter_img_new_dimensions[1] -= 1
            letter_img = cv2.resize(letter_img, tuple(letter_img_new_dimensions))
            show_frame[
            self.resolution[1] // 2 - letter_img.shape[0] // 2:
            self.resolution[1] // 2 + letter_img.shape[0] // 2,
            self.window_center[0] + self.window_size:
            self.window_center[0] + self.window_size + 2 * letter_img.shape[1] // 2,
            :] = letter_img

        # crops the area in the rectangle
        save_frame = save_frame[self.window_center[1] - self.window_size: self.window_center[1] + self.window_size,
                     self.window_center[0] - self.window_size: self.window_center[0] + self.window_size]

        show_frame = np.array(show_frame)

        return show_frame, save_frame

    def save_video(self, array):
        out = cv2.VideoWriter(join(self.samples_path, f'{int(time.time())}_{self.mimed_letters[-1]}.mp4'),
                              cv2.VideoWriter_fourcc(*'mp4v'), self.n_frames,
                              (array.shape[2], array.shape[1]), True)
        for f in array:
            out.write(f)
        out.release()

    def capture_frame(self):
        prev_frame_time, new_frame_time = 0, 0
        self.state_starting_time = time.time()

        saved_frames = []
        while self.is_running:
            current_time = time.time()
            # waits for the state to change
            if current_time >= self.state_starting_time + \
                    self.states_graph.get_seconds(self.states_graph.current_state):
                self.states_graph.set_current_state(self.states_graph.states[self.states_graph.current_state]["to"])
                self.state_starting_time, self.states_graph.is_new_state = current_time, True
            else:
                self.states_graph.is_new_state = False

            try:
                ret, frame = self.vid.read()
                show_frame, save_frame = self.frame_elaboration(frame,
                                                                horizontal_flip=False)
            except Exception as exception:
                print(exception)
                break

            if self.states_graph.is_new_state and self.states_graph.current_state == "idle":
                frames_to_save = np.stack(saved_frames)[np.linspace(0, len(saved_frames), num=int(self.seconds_to_be_recorded * self.n_frames),
                                                                    endpoint=False, dtype=int)]
                self.save_video(frames_to_save)
                saved_frames = []

            elif self.states_graph.current_state == "recording":
                saved_frames += [save_frame]
            # saves the frame in memory
            # save_frame = cv2.cvtColor(save_frame, cv2.COLOR_BGR2GRAY)
            # self.q_frames[:-1] = self.q_frames[1:]
            # self.q_frames[-1] = save_frame

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


class CameraStatesGraph:

    def __init__(self):
        self.states = {}
        self.current_state = None
        self.is_new_state = True

    def add_state(self, name, seconds):
        assert isinstance(name, str)
        assert seconds > 0
        if name in self.states.keys():
            return
        self.states[name] = {
            "to": None,
            "seconds": seconds
        }

    def add_edge(self, edge_from, edge_to):
        assert {edge_from, edge_to}.issubset(set(self.states.keys()))
        self.states[edge_from]["to"] = edge_to

    def set_current_state(self, state):
        assert isinstance(state, str)
        assert state in self.states
        self.current_state = state

    def get_seconds(self, state):
        assert isinstance(state, str)
        assert state in self.states
        return self.states[state]["seconds"]

    def __str__(self):
        string = f"Graph:\n{pformat(self.states, indent=4)}\n"
        string += f"Current state: {self.current_state}"
        return string
