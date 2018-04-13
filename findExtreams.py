#import the necessary packs
import imutils
import cv2

def get_extreams(image):
    print("Entered find extreams method..")
    ## load the image nd convert it grey sclae and blur it a bit
    #image = cv2.imread("/home/gch/Desktop/hand.jpg")
    #image = cv2.imread("/home/gch/Desktop/grabcut_out/grabcutout_cv2.png")
    #print("Image in findExtreams - ", image)
    ##resizing the image...!!
    #image = cv2.resize(image, (916, 1200))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    #thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # show the output image
    cv2.imshow("Applied Erosion and Dilations", thresh)
    cv2.waitKey(0)

    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    #Before we can find extreme points along a contour, it’s #important to understand that a contour is simply a NumPy #array of (x, y)-coordinates. Therefore, we can leverage #NumPy functions to help us find the extreme coordinates.
    
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    
    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
    cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    cv2.circle(image, extBot, 8, (255, 255, 0), -1)
    ##Show the output image
    cv2.imshow("img_extreams", image)
    cv2.waitKey(0)
    cv2.imwrite("/home/gch/Desktop/temp_out/extreamPts.jpg", image)
    
    ## draw bbox
    cv2.rectangle(image, (extLeft[0], extTop[1]), (extRight[0], extBot[1]), (255,0,0), 2)
    ##Show the output image
    cv2.imshow("bbox_image", image)
    cv2.waitKey(0)
    cv2.imwrite("/home/gch/Desktop/temp_out/extreamPtsWithBbox.jpg", image)
    
    print("image shape _ ", image.shape)
    print("The extream points are ... ")
    print("(extLeft, extTop) - ({:d}, {:d})".format(extLeft[0], extTop[1]))
    print("(extRight, extBot) - ({:d}, {:d})".format(extRight[0], extBot[1]))
    print("extLeft - ", extLeft)
    print("extRight - ", extRight)
    print("extTop - ", extTop)
    print("extBot - ", extBot)
    return [(extLeft[0], extTop[1]), (extRight[0], extBot[1])]
    
