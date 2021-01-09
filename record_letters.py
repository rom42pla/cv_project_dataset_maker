from modules.camera import Camera
import time
import argparse

# parses the arguments from the console
parser = argparse.ArgumentParser(description='Parameters for alphabet sign recognition')
parser.add_argument("-c", "--cam_number", type=int, default=0,
                    help='Number of the cam in use')
parser.add_argument("-f", "--n_frames", type=int, default=30,
                    help='Frames to be recorded')
parser.add_argument("-s", "--sleep_time", type=int, default=500,
                    help='Seconds for testing')
parser.add_argument("-r", "--resolution", type=int, default=360,
                    help='Resolution (width) of the webcam')
parser.add_argument("-d", "--show_fps", dest='show_fps', action='store_true',
                    help='Whether to show FPS or not')
args = parser.parse_args()

# debug mode
camera = Camera(cam_number=args.cam_number, n_frames=args.n_frames, resolution=args.resolution, show_fps=args.show_fps)

camera.start()

time.sleep(args.sleep_time)

camera.stop()

# c.plot_last_frames()
