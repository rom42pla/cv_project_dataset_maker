import re
from os import listdir
from os.path import join, exists, isdir, splitext

import torchvision


def load_dataset(videos_path: str):
    assert isinstance(videos_path, str)
    assert exists(videos_path) and isdir(videos_path)
    dataset = []
    for video_filename in listdir(videos_path):
        # ignores non-videos
        if not re.fullmatch(pattern=r"[0-9]*\_([a-z]|[A-Z])\.mp4", string=video_filename):
            continue
        # extracts the video and the label
        video, label = torchvision.io.read_video(join(videos_path, video_filename), pts_unit="sec")[0] / 255, \
                       splitext(video_filename)[0].split("_")[1]
        # adds the infos to the dataset
        dataset += [(video, label)]
    return dataset
