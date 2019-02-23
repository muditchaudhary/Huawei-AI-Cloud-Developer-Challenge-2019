import shutil, os

metafile = open('meta/small_labels_25c.txt')
target = 'small_sample/'
source = 'images/'
for line in metafile:
    dest = target + line.rstrip('\n')
    src = source + line.rstrip('\n')
    shutil.move(src, dest)

