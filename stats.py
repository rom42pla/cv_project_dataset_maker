import re
from os import listdir
from os.path import join, exists, isdir, splitext

import matplotlib.pyplot as plt

import torchvision


def load_dataset(samples_path: str):
    assert isinstance(samples_path, str)
    assert exists(samples_path) and isdir(samples_path)
    dataset = []
    for take in listdir(samples_path):
        if isdir(join(samples_path, take)):
            for sample_name in listdir(join(samples_path, take)):
                # ignores non-videos
                if not re.fullmatch(pattern=r"([a-z]|[A-Z])\.mp4", string=sample_name):
                    continue
                # extracts the video and the label
                video, label = torchvision.io.read_video(join(samples_path, take, sample_name), pts_unit="sec")[
                                   0] / 255, \
                               splitext(sample_name)[0]
                # adds the infos to the dataset
                dataset += [(video, label)]
    return dataset


dataset = load_dataset(join("assets", "samples"))

# labels distribution
labels = [label for _, label in dataset]
labels_unique = sorted(list(set(labels)), reverse=True)
sizes = [labels.count(label) for label in labels_unique]

fig1, ax = plt.subplots()
ax.pie(sizes, labels=labels_unique, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')
plt.title(f"Labels distribution for {len(labels)} videos")
plt.tight_layout()
plt.show()
