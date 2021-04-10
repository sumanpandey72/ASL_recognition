<<<<<<< HEAD
from PIL import Image
import os, sys

path = "C:/Users/suman/Desktop/test_dataset/asl_alphabet_train/asl_alphabet_train/B/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((28,28), Image.ANTIALIAS)
            imResize.save(f + 'r.jpg', 'JPEG', quality=90)

=======
from PIL import Image
import os, sys

path = "C:/Users/suman/Desktop/test_dataset/asl_alphabet_train/asl_alphabet_train/B/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((28,28), Image.ANTIALIAS)
            imResize.save(f + 'r.jpg', 'JPEG', quality=90)

>>>>>>> 5655fa7d12e3a971a4f5d4c17ff968dbedcc5a5a
resize()