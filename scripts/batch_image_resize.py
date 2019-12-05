# Batch image resizer
from PIL import Image

import os

path = 'C:\\Users\\fc5ntb\\Downloads\\iCloud Photos'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpeg' in file.lower():
            files.append(os.path.join(r, file))


basewidth = 300
i = 42
for f in files:
    print(f)
    img = Image.open(f)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('../dooble_pics/TO_LABEL/dooble{}.jpg'.format(i))
    i = i + 1
