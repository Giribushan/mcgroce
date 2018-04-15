import numpy as np
import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import findExtreams as fe
import os
import random
import copy
#from curses.textpad import rectangle

def addAlpha(img):
    # Adding the alpha channel
    tmp2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha2 = cv2.threshold(tmp2,0,255,cv2.THRESH_BINARY) 
    b, g, r = cv2.split(img) 
    rgba = [b,g,r, alpha2] 
    img = cv2.merge(rgba,4)
    return img

# bbox coordinates
bbox_coordinates = []
# change the name as per the product 
product_name = "ChocolateCake"
product_name = "tataSalt"

# Object folder path
#objFolderPath = "C:\\Users\\gcheru200\\Pictures\\chocCake"
#objFolderPath = "C:\\Users\\gcheru200\\Pictures\\stabCake"
objFolderPath = "C:\\Users\\gcheru200\\Pictures\\tts"
#Background Images folder path
bgFolderPath = "C:\\Users\\gcheru200\\Pictures\\bg_selected"

#bgFgSetSize is to set many number of diverse (background+Foreground) as you want in the training
bgFgSetSize = 5

## variation_size is to specify as many number of variations as you want for a single (Foreground+Background)
#variation_size = 300
variation_size = 2

# flag to reuse the same bg, for multiple variations
reuseSameBG = True

for j in range(bgFgSetSize):
    i = 0
    bbox_coordinates = []
    objImagePath = os.path.join(objFolderPath,random.choice(os.listdir(objFolderPath)))
    bgImagePath = os.path.join(bgFolderPath,random.choice(os.listdir(bgFolderPath)))
    #This is the input image with object alone..
    inputObj = cv2.imread(objImagePath)
    #cv2.imshow('input image', inputObj)
    #cv2.waitKey(0)
    
    # Perfrom edge detection
    img = cv2.Canny(inputObj, 100, 255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    #print("The edges shape - ", img.shape)
    #imgplot = plt.imshow(img)
    #plt.show()
    bbox_coordinates = fe.get_extreams(img)
    
    #Using these coordinates to initialize the rectangle for grabcut
    rect = (bbox_coordinates[0][0], bbox_coordinates[0][1], bbox_coordinates[1][0] - bbox_coordinates[0][0], bbox_coordinates[1][1] - bbox_coordinates[0][1])
    
    #print("Input image shape - ", img.shape)
    #imgplot = plt.imshow(cv2.cvtColor(inputObj, cv2.COLOR_BGR2RGBA))
    # imgplot = plt.imshow(img)
    #plt.show()
    
    
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    #rect = (50,50,450,290)
    #rect = (1, 1, img.shape[1], img.shape[0])
    cv.grabCut(inputObj,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    inputObj = inputObj*mask2[:,:,np.newaxis]
    #cv2.imshow("grabcutOut", inputObj)
    #cv2.waitKey(0)
    
    # Adding the alpha channel to Canny edge image
    img = addAlpha(img)
    ## Adding alpha channel to Original image
    inputObj = addAlpha(inputObj)
    
    #cv2.imshow("GrabCut Output", inputObj)
    #cv2.waitKey(0)
    #print("The alpha channel dimentions..", img[:,:,3].shape)
    
    ## Now we are going to get the bbox coordinates as per the grabcut output
    bbox_coordinates = fe.get_extreams(img)
    # debugging loggers
    #print('bbox_coordinates - ' ,bbox_coordinates)
    
    #print("Image dimensions with out extreams - ", img.shape)
    ## Cropping the images as per the extream points..
    inputObj = inputObj[bbox_coordinates[0][1]:bbox_coordinates[1][1],bbox_coordinates[0][0]:bbox_coordinates[1][0]]
    #img = img[bbox_coordinates[0][1]:bbox_coordinates[1][1],bbox_coordinates[0][0]:bbox_coordinates[1][0]]
    #print("Image dimensions with extreams - ", img.shape)
    
    #plt.imshow(inputObj)
    #plt.show()
    #cv2.waitKey(0)
    
    ## Now we are going to overlay the object on top of background image..
    #background = cv2.imread("C:\\Users\\gcheru200\\Pictures\\others\\big-data-binary.png")
    #background = cv2.imread("C:\\Users\\gcheru200\\Pictures\\others\\big-data-binary.png", cv2.IMREAD_UNCHANGED)
    #print("The background image to be read - ", bgImagePath)
    background = cv2.imread(bgImagePath, cv2.IMREAD_UNCHANGED)
    freshBackGround = cv2.imread(bgImagePath, cv2.IMREAD_UNCHANGED)
    #imgplot = plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    ## Calculate the threshold to sample the input object positions w.r.to background
    xlimit = background.shape[0] - inputObj.shape[0]
    ylimit = background.shape[1] - inputObj.shape[1]
    # debugging loggers
#     print("The input object width - ", inputObj.shape[0])
#     print("The input object height - ", inputObj.shape[1])
#     print("Background width - ", background.shape[0])
#     print("Background height - ", background.shape[1])
#     print("The xlimit value - ", xlimit)
#     print("The ylimit value - ", ylimit)
    
    
    
    ## Now Overlaying the product on top of a background image..
    import random
    import overlay_image as oi
    import matplotlib.pyplot as plt
    import xml.etree.cElementTree as ET
    alpha = inputObj[:, :, 3]
    if variation_size > 1:
        stackObjAnnts = True
    else:
        stackObjAnnts = False
    #objTemp = null
    objTemp = ET.SubElement(ET.Element("annotation"), "object")
    for i in range(variation_size):
        ## picking the random coordinates with in the background range..
        #i += j
        print("i value - ", str(i))
        print("j value - ", str(j)) 
        if variation_size > 1 and reuseSameBG:
            # reusing the same background
            background = background
        elif variation_size > 1 and not reuseSameBG:
            # read the fresh background
            freshBackGround = cv2.imread(bgImagePath, cv2.IMREAD_UNCHANGED)
            background = freshBackGround
            cv2.imshow("The bg", freshBackGround)
            cv2.waitKey(0)
            
        offset_x = random.randint(0,xlimit)
        offset_y = random.randint(0,ylimit) 
        # debugging loggers
        #print("The offset x - ", offset_x)
        #print("The offset y - ", offset_y)
        # Overlaying the object with background image..Kind of alpha blending..
        # So the background image will be manipulated and can be used for training!!!!
        oi.overlay_image_alpha(background,
                        inputObj[:, :, 0:4],
                        (offset_x, offset_y),
                        alpha )
        #cv2.imshow('background blended with foreground', background)
        #cv2.waitKey(0)
    
        # Global bbox coordinates with respect to background
        global_x1 = offset_y
        global_y1 = offset_x 
        global_x2 = inputObj.shape[1] + offset_y
        global_y2 = inputObj.shape[0] + offset_x
        
        
        # debugging loggers
        #print("The global bbox coordinates w.r.to background..")
        #print(global_x1,global_y1)
        #print(global_x2,global_y2)
        ## draw bbox on final image to get a final picture
        FinalImage_with_bbox = copy.copy(background)
        FinalImage_with_bbox = cv2.rectangle(FinalImage_with_bbox, (global_x1, global_y1), (global_x2, global_y2), (0,255,0), 2)
    
    #     plt.imshow(cv2.cvtColor(FinalImage_with_bbox, cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    #     plt.show()
        ##Show the output image
        #cv2.imshow("FinalImage_with_bbox ", FinalImage_with_bbox )
        #cv2.waitKey(0)
        cv2.imwrite('C:\\Users\\gcheru200\\Pictures\\others\\Images\\' + product_name + "_fb" + str(j)+ "_v" + str(i) + ".jpg", background)
        #print("Saved training image as => " + 'C:\\Users\\gcheru200\\Pictures\\others\\Images\\' + product_name + "_" + str(i) + ".jpg")
    
        ## Now we are going to parse the bbox coordinates to Pascal VOC annotation format.
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "folder"
        fn = ET.SubElement(root, "filename").text = product_name + "_fb" + str(j)+ "_v" + str(i) + ".jpg"
        ET.SubElement(root, "path").text = "path"
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(background.shape[0])
        ET.SubElement(size, "height").text = str(background.shape[1])
        ET.SubElement(size, "depth").text = str(3)
        ET.SubElement(root, "segmented").text = str(0)
        obj = ET.SubElement(root, "object")
        if i !=  0 and reuseSameBG:
            root.append(objTemp)
        
        ET.SubElement(obj, "name").text = product_name 
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbx = ET.SubElement(obj, "bndbox")
    
        ET.SubElement(bbx, "xmin").text = str(global_x1)
        ET.SubElement(bbx, "ymin").text = str(global_y1)
        ET.SubElement(bbx, "xmax").text = str(global_x2)
        ET.SubElement(bbx, "ymax").text = str(global_y2)
        if reuseSameBG:
            objTemp = copy.copy(obj)
    
        tree = ET.ElementTree(root)
        tree.write("C:\\Users\\gcheru200\\Pictures\\others\\Annotaions\\" + product_name + "_" + "fb" + str(j)+ "_v" + str(i) + ".xml")
        # debugging loggers !!!
        #print("Saved annotation as => " + "C:\\Users\\gcheru200\\Pictures\\others\\Annotaions\\" + product_name + "_" + str(i) + ".xml")
    cv2.destroyAllWindows()
print("Completed..")
    
'''
#cv2.imwrite("/home/gch/Desktop/grabcut_out/backandwhite_mask.png", img)
#import scipy.misc
#print("THe shape of the mask = ", img.shape)
import PIL
from PIL import Image
im = Image.fromarray(img)
print("The image mode - ", im.mode)
im = im.convert("RGBA")
print("The image mode after convertion - ", im.mode)
im.show()
im.save("/home/gch/Desktop/grabcut_out/backandwhite_mask.png")
#scipy.misc.imsave("/home/gch/Desktop/grabcut_out/backandwhite_mask.jpg", img)
#cv2.imshow('',img)
#cv2.waitKey(0)
#plt.imshow(img),plt.colorbar(),plt.show()

'''
'''
# newmask is the mask image I manually labelled
newmask = cv2.imread('newmask.png',0)

# whereever it is marked white (sure foreground), change mask=1
# whereever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1

mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
'''
