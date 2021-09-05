#pbtext_to_text.py
import json

with open('label_map.pbtxt') as f:
    txt = f.read()
    splitted  = txt.split ()
    print(splitted)
    if splitted[0] == 'name':
        print(splitted[1])
