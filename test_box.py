import PIL
from PIL import Image

mywidth = 28

img = Image.open('Output.jpg')
wpercent = (mywidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
img.save('resized_output.jpg')