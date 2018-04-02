import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

#img = cv2.imread('messi5.jpg')
#img = cv.imread('/home/gch/Desktop/ObjectWithWhiteBG/output.jpg')
img = cv.imread('/home/gch/Desktop/goodday.jpg')
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#rect = (50,50,450,290)
rect = (10,10,450,290)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


tmp2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,alpha2 = cv2.threshold(tmp2,0,255,cv2.THRESH_BINARY) 
b, g, r = cv2.split(img) 
rgba = [b,g,r, alpha2] 
img = cv2.merge(rgba,4)
cv2.imwrite("/home/gch/Desktop/grabcut_out/grabcutout_cv2.png", img)
cv2.imshow("GrabCut Output", img)
cv2.waitKey(0)
print("The alpha channel dimentions..", img[:,:,3].shape)

## Now we are going to get the bbox coordinates as per the mask
bbox_coordinates = []
import findExtreams as fe
bbox_coordinates = fe.get_extreams(img)
print('bbox_coordinates - ' ,bbox_coordinates)


## Now we are going to overlay the object on top of background image..
background = cv2.imread("/home/gch/Desktop/Somali.png", cv2.IMREAD_UNCHANGED)
alpha = img[:, :, 3]
import random
print(random.randint(0,9))
print("The complete bbox coordinates..")
print(bbox_coordinates[1][0])
print(bbox_coordinates[0][0])
print(bbox_coordinates[1][1])
print(bbox_coordinates[0][1])

xlimit = background.shape[0] - (bbox_coordinates[1][0] - bbox_coordinates[0][0])
ylimit = background.shape[1] - (bbox_coordinates[1][1] - bbox_coordinates[0][1])
print("The xlimit value - ", xlimit)
print("The ylimit value - ", ylimit)

## picking the random coordinates with in the background range..
import random
offset_x = random.randint(0,xlimit)
offset_y = random.randint(0,ylimit) 
print("The offset coordinates..")
print("The offset x - ", offset_x)
print("The offset y - ", offset_y)
import overlay_image as oi
oi.overlay_image_alpha(background,
                    img[:, :, 0:4],
                    (offset_x, offset_y),
                    alpha )
cv2.imshow('backgrund with foreground', background)
cv2.waitKey(0)

print("the coords...")

global_x1 = bbox_coordinates[0][0] + offset_y 
global_y1 = bbox_coordinates[0][1] + offset_x
global_x2 = bbox_coordinates[1][0] + offset_y
global_y2 = bbox_coordinates[1][1] + offset_x
'''
global_x1 = bbox_coordinates[0][0]  
global_y1 = bbox_coordinates[0][1] 
global_x2 = bbox_coordinates[1][0] 
global_y2 = bbox_coordinates[1][1] 
'''
print("The global coordinates..")
print(global_x1,global_y1)
print(global_x2,global_y2)
## draw bbox on final image
background = cv2.rectangle(background, (global_x1, global_y1), (global_x2, global_y2), (255,0,0), 2)
##Show the output image
#cv2.imshow("bbox_image", background)
#cv2.waitKey(0)
print("Going to visualize using matpltlib..")
import matplotlib.pyplot as plt
imgplot = plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.show()
#plt.savefig('/home/gch/Desktop/Final_Images_and_annotations/Images/systhesized_out.jpg', background)
cv2.imwrite('/home/gch/Desktop/Final_Images_and_annotations/Images/systhesized_out.jpg', background)
print("Complted ..")
#import PIL 
#from PIL import Image
#background = Image.fromarray(background)
#background.show("The final image")


## writing to annotations..
import xml.etree.cElementTree as ET

root = ET.Element("annotation")
ET.SubElement(root, "filename").text = 0
ET.SubElement(root, "folder").text = 0
ET.SubElement(root, "path").text = 0
ET.SubElement(root, "source").text = 0
ET.SubElement(root, "size").text = 0
ET.SubElement(root, "segmented").text = 0
obj = ET.SubElement(root, "object")

ET.SubElement(obj, "name").text = 0
ET.SubElement(obj, "pose").text = 0
ET.SubElement(obj, "truncated").text = 0
ET.SubElement(obj, "difficult").text = 0
bbx = ET.SubElement(obj, "bndbox")

#ET.SubElement(bbx, "xmin").int = global_x1
#ET.SubElement(bbx, "ymin").int = global_y1
#ET.SubElement(bbx, "xmax").int = global_x2
#ET.SubElement(bbx, "ymax").int = global_y2

#ET.SubElement(doc, "field1", name="blah").text = "some value1"

tree = ET.ElementTree(root)
tree.write("/home/gch/Desktop/Final_Images_and_annotations/Annotaions/annotation_out.xml")


