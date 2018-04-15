import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

def addAlpha(img):
    # Adding the alpha channel
    tmp2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha2 = cv2.threshold(tmp2,0,255,cv2.THRESH_BINARY) 
    b, g, r = cv2.split(img) 
    rgba = [b,g,r, alpha2] 
    img = cv2.merge(rgba,4)
    return img

# change the name as per the product 
product_name = "ChocolateCake"

#img = cv2.imread("C:\\Users\\gcheru200\\Pictures\\others\\image4.jpg")

#This is the original input image
img2 = cv2.imread("C:\\Users\\gcheru200\\Pictures\\chocCake\\ChocCake1.png")
# Perfrom edge detection
img2WithEdges = cv2.Canny(img2, 0, 255)
cv2.imwrite("C:\\Users\\gcheru200\\Pictures\\others\\ChocCakeWithEdges.jpg", img2WithEdges)

#This image is from edgeDetection
img = cv2.imread("C:\\Users\\gcheru200\\Pictures\\others\\ChocCakeWithEdges.jpg")

print("Input image shape - ", img.shape)
#img = cv2.imread("C:\\Users\\gcheru200\\Pictures\\yeppie\\yeppie1.png")
# Plot the input image
#imgplot = plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGBA))
imgplot = plt.imshow(img)
plt.show()

'''
##replacing the white bg pixels to pure white pixels..
#r1, g1, b1 = range(165,180), range(165,180), range(169,199) # Original value
r1, g1, b1 = 0, 0, 0 # Value that we want to replace it with
r2, g2, b2 = 255,255,255 # Original value
red, green, blue= img[:,:,0], img[:,:,1], img[:,:,2]
qmask = (red == r1) & (green == g1) & (blue == b1)
#mask = (np.any(red >= r1[0]) and np.any(red <= r1[1])) and (np.any(green >= g1[0]) and np.any(green <= g1[1])) and (np.any(blue >= b1[0]) and np.any(blue <= b1[1]))
#mask = (red >= r1[0]).all() and (red <= r1[1]).all() & ((green >= r2[0]).all() and (green <= r2[1]).all())
#mask = ((red >= r1[0]).any() and (red <= r1[1]).any()) & ((green >= r2[0]).any() and (green <= r2[1]).any()) & ((blue >= r2[0]).any() and (blue <= r2[1]).any())
mask = (red in r1).all() & (green in g1).all() & (blue in b1).all()
img[:,:,:3][mask] = [r2, g2, b2]
print("The targeted pixels..", img[:,:,:3][mask])
## after changeing to pure white background..
imgplot = plt.imshow(img[:,:,:3][mask])
plt.show()
'''

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#Initialize the rectangle, 
# Try the rectangle which will enclose the object. 
#rect = (34,370,340,700)
##Apply rectangle up to the size of input image

# Initialize the rectangle to desired area which can minimize the grabcut time..
#rect = (1, 1, img.shape[1], img.shape[0])
rect = (490, 222, 804, 470)
##Apply grab cut..
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#mask2 = np.where((mask==3),255,0).astype('uint8')
img = img*mask2[:,:,np.newaxis]
img2 = img2*mask2[:,:,np.newaxis]

# Adding the alpha channel to canny edge image
img = addAlpha(img)
## Adding alpha channel to original image
img2 = addAlpha(img2)
cv2.imwrite("/home/gch/Desktop/grabcut_out/grabcutout_cv2.png", img)
import PIL 
from PIL import Image
imgPil = Image.fromarray(img2)
print("The image pil - ", imgPil)
#cv2.imshow("GrabCut Output", cv2.cvtColor(img2, cv2.COLOR_BGRA2RGB))
cv2.imshow("GrabCut Output", img2)
cv2.waitKey(0)
print("The alpha channel dimentions..", img[:,:,3].shape)

## Now we are going to get the bbox coordinates as per the mask
bbox_coordinates = []
import findExtreams as fe
bbox_coordinates = fe.get_extreams(img)
print('bbox_coordinates - ' ,bbox_coordinates)

print("Image dimensions with out extreams - ", img.shape)

## Draw the bbox on top of original image...
img = img2[bbox_coordinates[0][1]:bbox_coordinates[1][1],bbox_coordinates[0][0]:bbox_coordinates[1][0]]
print("Image dimensions with extreams - ", img.shape)
plt.imshow(img)
plt.show()
#cv2.waitKey(0)


## Now we are going to overlay the object on top of background image..
#background = cv2.imread("/home/gch/Desktop/storeHD.jpg", cv2.IMREAD_UNCHANGED)
background = cv2.imread("C:\\Users\\gcheru200\\Pictures\\others\\big-data-binary.png")

alpha = img[:, :, 3]
import random
print(random.randint(0,9))
print("The complete bbox coordinates..")
print(bbox_coordinates[1][0])
print(bbox_coordinates[0][0])
print(bbox_coordinates[1][1])
print(bbox_coordinates[0][1])

## Calculate the threshold to sample the input object with in the background boundaries
xlimit = background.shape[0] - img.shape[0]
ylimit = background.shape[1] - img.shape[1]
print("The input object width - ", img.shape[0])
print("The input object height - ", img.shape[1])
print("Background width - ", background.shape[0])
print("Background height - ", background.shape[1])
print("The xlimit value - ", xlimit)
print("The ylimit value - ", ylimit)
# The background image used here
imgplot = plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.show()


## Now Overlaying the product on top of a background image..
import random
import overlay_image as oi
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET

## Specify the dataset_size
#dataset_size = 300
dataset_size = 1
for i in range(dataset_size):
    ## picking the random coordinates with in the background range..
    offset_x = random.randint(0,xlimit)
    offset_y = random.randint(0,ylimit) 
    # debugging loggers
    print("The offset x - ", offset_x)
    print("The offset y - ", offset_y)
    background = cv2.imread("C:\\Users\\gcheru200\\Pictures\\others\\big-data-binary.png", cv2.IMREAD_UNCHANGED)
    # Overlaying the object with background image..Kind of alpha blending..
    oi.overlay_image_alpha(background,
                    img[:, :, 0:4],
                    (offset_x, offset_y),
                    alpha )
    cv2.imshow('background blended with foreground', background)
    cv2.waitKey(0)

    # Global coordinates with respect to background
#     global_x1 = bbox_coordinates[0][0]  
#     global_y1 = bbox_coordinates[0][1] 
#     global_x2 = bbox_coordinates[1][0] + offset_y
#     global_y2 = bbox_coordinates[1][1] + offset_x

    global_x1 = offset_y
    global_y1 = offset_x 
    global_x2 = img.shape[1] + offset_y
    global_y2 = img.shape[0] + offset_x
    
    
    # debugging loggers
    print("The global coordinates w.r.to background..")
    print(global_x1,global_y1)
    print(global_x2,global_y2)
    
    ## draw bbox on final image to get a final picture
    FinalImage_with_bbox = cv2.rectangle(background, (global_x1, global_y1), (global_x2, global_y2), (0,0,255), 2)

    plt.imshow(cv2.cvtColor(FinalImage_with_bbox, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.show()
    ##Show the output image
    cv2.imshow("FinalImage_with_bbox ", FinalImage_with_bbox )
    cv2.waitKey(0)
    cv2.imwrite('/home/gch/Desktop/Final_Images_and_annotations/Images/' + product_name + "_" + str(i) + ".jpg", background)
    print("Completed saving the training image.." + product_name + "_" + str(i) + ".jpg")


    ## Now we are going to parse the bbox coordinates to Pascal VOC annotation format.
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "folder"
    ET.SubElement(root, "filename").text = product_name + "_" + str(i) + ".jpg"
    ET.SubElement(root, "path").text = "path"
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(background.shape[0])
    ET.SubElement(size, "height").text = str(background.shape[1])
    ET.SubElement(size, "depth").text = str(3)
    ET.SubElement(root, "segmented").text = str(0)
    obj = ET.SubElement(root, "object")

    ET.SubElement(obj, "name").text = product_name 
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = str(0)
    ET.SubElement(obj, "difficult").text = str(0)
    bbx = ET.SubElement(obj, "bndbox")

    ET.SubElement(bbx, "xmin").text = str(global_x1)
    ET.SubElement(bbx, "ymin").text = str(global_y1)
    ET.SubElement(bbx, "xmax").text = str(global_x2)
    ET.SubElement(bbx, "ymax").text = str(global_y2)


    tree = ET.ElementTree(root)
    tree.write("C:\\Users\\gcheru200\\Pictures\\others\\Annotaions\\" + product_name + "_" + str(i) + ".xml")
    # debugging loggers !!!
    print("Completed saving the annotation.." + product_name + "_" + str(i) + ".xml")
print("Completed..")
