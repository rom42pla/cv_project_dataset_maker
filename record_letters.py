import argparse
from os.path import join

from modules.camera import Camera

assets_path = join(".", "assets")
samples_path, letters_path = join(assets_path, "samples"), join(assets_path, "letters")

# parses the arguments from the console
parser = argparse.ArgumentParser(description='Parameters for Alphabet Sign Recognition')
parser.add_argument("-c", "--cam_number", type=int, default=0,
                    help='Number of the cam in use')
parser.add_argument("-sr", "--seconds_to_be_recorded", type=float, default=2,
                    help='Seconds to be recorded')
parser.add_argument("-si", "--seconds_to_be_idle", type=float, default=6,
                    help='Seconds to be idle')
parser.add_argument("-fr", "--n_frames", type=int, default=30,
                    help='Frames to be recorded')
parser.add_argument("-r", "--resolution", type=int, default=720,
                    help='Resolution (width) of the webcam')
parser.add_argument("-d", "--show_fps", dest='show_fps', action='store_true',
                    help='Whether to show FPS or not')
parser.add_argument("-a", "--assets_path", type=str, default=assets_path,
                    help='Path of the assets')
args = parser.parse_args()

camera = Camera(assets_path=args.assets_path,
                cam_number=args.cam_number, resolution=args.resolution, show_fps=args.show_fps,
                n_frames=args.n_frames,
                seconds_to_be_idle=args.seconds_to_be_idle, seconds_to_be_recorded=args.seconds_to_be_recorded)

camera.start()
