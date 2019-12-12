from PIL import Image

"""

_____________________
|                   |
|                   |
|                   |
|                   |
_____________________

"""
def horizontal_split(img_path, label_path, y):
    im = Image.open(img_path)
    imgwidth, imgheight = im.size

    im1 = im.crop((0, 0, imgwidth, y)) 
    im2 = im.crop((0, y, imgwidth, imgheight)) 
  
    # Shows the image in image viewer 
    im1.show() 
    im2.show()

    """
    try:
        o = a.crop(area)
        o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
    except:
        pass
    """



file_path = r"D:\TensorFlow\private_project\dooble_pics\TO_LABEL\dooble60.jpeg"
label_path = r"D:\TensorFlow\private_project\dooble_pics\TO_LABEL\dooble60.xml"

horizontal_split(file_path,label_path, 500)