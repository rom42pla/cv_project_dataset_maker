from os import makedirs, listdir, remove
from os.path import exists, join
import re
import string

from pprint import pformat

from threading import Thread
import cv2
import time
import numpy as np


class Camera:

    def __init__(self, assets_path: str,
                 cam_number: int = 0, resolution: int = 720, show_fps: bool = True,
                 n_frames: int = 30,
                 seconds_to_be_recorded: float = 2, seconds_to_be_idle: float = 6,
                 window_size: int = 175,
                 window_preparation_color: tuple = (255, 255, 255),
                 window_recording_color: tuple = (0, 0, 255),
                 window_idle_color: tuple = (0, 255, 0),
                 window_name: str = 'ASL Dataset Maker'):
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
        assert isinstance(window_name, str)
        self.window_name = window_name

        # states' variables
        self.mimed_letters = []
        self.takes = [str(int(time.time()))]
        self.alphabet = list(string.ascii_lowercase)
        self.state_starting_time = None
        self.seconds_to_be_recorded, self.seconds_to_be_idle = seconds_to_be_recorded, \
                                                               seconds_to_be_idle

        # creates the states' graph
        self.states_graph = CameraStatesGraph()
        self.states_graph.add_state("preparation", seconds=None,
                                    window_color=window_preparation_color)
        self.states_graph.add_state("idle", seconds=self.seconds_to_be_idle,
                                    window_color=window_idle_color)
        self.states_graph.add_state("recording", seconds=self.seconds_to_be_recorded,
                                    window_color=window_recording_color)

        self.states_graph.add_edge(edge_from="preparation", edge_to="idle")
        self.states_graph.add_edge(edge_from="idle", edge_to="recording")
        self.states_graph.add_edge(edge_from="recording", edge_to="idle")

        self.states_graph.set_current_state("preparation")

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
        assert window_size < self.resolution[0] // 2
        self.window_center = (self.resolution[0] // 2, self.resolution[1] // 2)

        # structures used to save the various frames
        self.n_frames, self.window_size = n_frames, window_size
        self.q_frames = np.zeros(shape=(n_frames, self.window_size * 2, self.window_size * 2, 3))

    def capture_frame(self):
        prev_frame_time, new_frame_time = 0, 0
        self.state_starting_time = time.time()

        saved_frames = []
        while self.is_running:
            current_time = time.time()
            # waits for the state to change
            if self.states_graph.get_seconds(self.states_graph.current_state):
                if current_time >= self.state_starting_time + \
                        self.states_graph.get_seconds(self.states_graph.current_state):
                    self.states_graph.set_current_state(self.states_graph.states[self.states_graph.current_state]["to"])
                    self.state_starting_time, self.states_graph.is_new_state = current_time, True
                else:
                    self.states_graph.is_new_state = False

            try:
                ret, frame = self.vid.read()
                show_frame, save_frame = self.frame_elaboration(frame,
                                                                horizontal_flip=True)
            except Exception as exception:
                print(exception)
                break

            if self.states_graph.current_state == "idle" and \
                    self.states_graph.is_new_state and len(saved_frames) > 0:
                frames_to_save = np.stack(saved_frames)[
                    np.linspace(0, len(saved_frames), num=int(self.seconds_to_be_recorded * self.n_frames),
                                endpoint=False, dtype=int)]
                self.save_video(frames_to_save)
                saved_frames = []
            elif self.states_graph.current_state == "recording":
                saved_frames += [save_frame]

            # eventually shows FPS on the app
            if self.show_fps:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(img=show_frame, text=f"FPS: {np.round(fps)}",
                            org=(0, 30), color=(0, 255, 0),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
            cv2.imshow(self.window_name, show_frame)

            pressed_key = cv2.waitKey(1)
            if pressed_key == ord(" "):
                if self.states_graph.current_state == "preparation":
                    self.states_graph.set_current_state("idle")
                    self.state_starting_time = time.time()
                elif self.states_graph.current_state == "idle":
                    self.states_graph.set_current_state("preparation")
                    self.state_starting_time = time.time()

            # repeat previous letter
            if pressed_key == ord("r"):
                if self.states_graph.current_state in {"preparation", "idle"} and (
                        len(self.mimed_letters) > 0 or len(self.takes) > 1):
                    # removes the latest letter
                    if len(self.mimed_letters) == 0:
                        # removes the last take
                        if exists(join(self.samples_path, f"take_{self.takes[-1]}")):
                            remove(join(self.samples_path, f"take_{self.takes[-1]}"))
                        self.takes = self.takes[:-1]
                        latest_letter = self.alphabet[-1]
                        self.mimed_letters = self.alphabet[:-1]
                    else:
                        latest_letter = self.mimed_letters[-1]
                        self.mimed_letters = self.mimed_letters[:-1]
                    remove(join(self.samples_path, f"take_{self.takes[-1]}", f"{latest_letter}.mp4"))
                    self.state_starting_time = time.time()
                    if self.states_graph.current_state == "preparation":
                        self.states_graph.set_current_state("idle")

            if pressed_key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) == 0:
                self.stop()

    def get_previous_letter(self):
        previous_letter_index = self.alphabet.index(self.mimed_letters[-1]) - 1
        if previous_letter_index < 0:
            previous_letter_index = -1
        return self.alphabet[previous_letter_index]

    def get_next_letter(self):
        # default value
        next_letter_index = 0
        if len(self.mimed_letters) > 0:
            next_letter_index = self.alphabet.index(self.mimed_letters[-1]) + 1
            if next_letter_index >= len(self.alphabet):
                next_letter_index = 0
        return self.alphabet[next_letter_index]

    def frame_elaboration(self, save_frame, horizontal_flip: bool = True):
        # horizontally flips the image
        if horizontal_flip:
            save_frame = np.flip(save_frame, axis=1)

        show_frame = np.copy(save_frame)

        # draws a rectangle on the app
        window_color = self.states_graph.get_window_color(self.states_graph.current_state)
        show_frame = cv2.rectangle(show_frame,
                                   (self.window_center[0] - self.window_size, self.window_center[1] - self.window_size),
                                   (self.window_center[0] + self.window_size, self.window_center[1] + self.window_size),
                                   color=window_color, thickness=2)

        upper_label = f""
        if self.states_graph.current_state in {"idle", "recording"}:
            # shows the new letter to mimic
            if self.states_graph.is_new_state and self.states_graph.current_state == "idle":
                self.mimed_letters += [self.get_next_letter()]

            if self.states_graph.current_state == "idle":
                upper_label = f"prepare letter '{self.get_next_letter()}'"
            elif self.states_graph.current_state == "recording":
                upper_label = f"recording letter '{self.get_next_letter()}'"

            # shows an image to use as example for the next letter
            letter_img = [img_name for img_name in listdir(self.letters_examples) if
                          re.match(f"{self.get_next_letter()}\..*", img_name)]
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

            # show a progress bar
            percentage = (time.time() - self.state_starting_time) / \
                         self.states_graph.get_seconds(self.states_graph.current_state)
            show_frame = cv2.rectangle(show_frame,
                                       (self.window_center[0] - self.window_size,
                                        self.window_center[1] - self.window_size),
                                       (self.window_center[0] - self.window_size + int(
                                           self.window_size * percentage * 2),
                                        25 + self.window_size),
                                       color=window_color, thickness=-1)

        # upper label
        cv2.putText(img=show_frame, text=upper_label,
                    org=(self.window_center[0] - self.window_size, self.window_center[1] - self.window_size - 20),
                    color=window_color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, thickness=1)

        # commands label
        cv2.putText(img=show_frame, text=f"ESC - quit    SPACE - pause/resume    r - repeat previous letter",
                    org=(0 + 20, self.resolution[1] - 20),
                    color=self.states_graph.get_window_color("preparation"),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)

        # state label
        cv2.putText(img=show_frame, text=self.states_graph.current_state,
                    org=(self.window_center[0] - self.window_size + 5, self.window_center[1] + self.window_size - 5),
                    color=window_color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)

        # crops the area in the rectangle
        save_frame = save_frame[self.window_center[1] - self.window_size: self.window_center[1] + self.window_size,
                     self.window_center[0] - self.window_size: self.window_center[0] + self.window_size]

        show_frame = np.array(show_frame)

        return show_frame, save_frame

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

    def save_video(self, array):
        # eventually creates the folder
        take_path = join(self.samples_path, f"take_{self.takes[-1]}")
        if not (exists(take_path)):
            makedirs(take_path)
        # saves the video
        out = cv2.VideoWriter(join(take_path, f"{self.mimed_letters[-1]}.mp4"),
                              cv2.VideoWriter_fourcc(*'mp4v'), self.n_frames,
                              (array.shape[2], array.shape[1]), True)
        for f in array:
            out.write(f)
        out.release()
        # eventually reset the counters
        if len(self.mimed_letters) > 0 and set(self.mimed_letters) == set(self.alphabet):
            self.takes += [str(int(time.time()))]
            self.mimed_letters = []


class CameraStatesGraph:

    def __init__(self):
        self.states = {}
        self.current_state = None
        self.is_new_state = True

    def add_state(self, name, seconds,
                  window_color: tuple = (0, 0, 0)):
        assert isinstance(name, str)
        assert [color for color in window_color if color < 256] is not []
        if name in self.states.keys():
            return
        self.states[name] = {
            "to": None,
            "seconds": seconds,
            "window_color": window_color
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

    def get_window_color(self, state):
        assert isinstance(state, str)
        assert state in self.states
        return self.states[state]["window_color"]

    def __str__(self):
        string = f"Graph:\n{pformat(self.states, indent=4)}\n"
        string += f"Current state: {self.current_state}"
        return string
