from pathlib import Path
from glob import glob
import collections
import os

path = Path('/mnt/c/Users/davi9801/Desktop/del')
flist =[]
for dir in path.iterdir():
    if os.path.isdir(dir):
        tmp_list = glob(os.path.join(dir, "*"))
        for file in tmp_list:
            filename = file.split("/")[-1][:-4]
            flist.append(filename)

dup_files = list()
for item, count in collections.Counter(flist).items():
    if count > 1:
        dup_files.append(item)

all_files = []
for dir in path.iterdir():
    if os.path.isdir(dir):
        tmp_list = glob(os.path.join(dir, "*"))
        for file in tmp_list:
            if file.split("/")[-1][:-4] not in dup_files:
                os.remove(file)
