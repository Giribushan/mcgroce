'''
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = ( alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                img[y1:y2, x1:x2, c])
'''


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
   #img_overlay = cv2.imread("smaller_image.png", -1)
   #y1, y2 = pos[0], pos[1] + img_overlay.shape[0]
   #x1, x2 = pos[0], pos[1] + img_overlay.shape[1]
   y1, y2 = pos[0], pos[0] + img_overlay.shape[0]
   x1, x2 = pos[1], pos[1] + img_overlay.shape[1]

   alpha_s = img_overlay[:, :, 3] / 255.0
   alpha_l = 1.0 - alpha_s
   ## bbox coordinates
   #lambda x: x
   ## x coordinates of pixels, whose alpha values not equals to zero
   ##x, y = pos

   #x1, x2 = max(0, x), min(alpha_mask.shape[1], x + img_overlay.shape[1])
   #y1, y2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

   for c in range(0, 3):
       img[y1:y2, x1:x2, c] = (alpha_s * img_overlay[:, :, c] +
                              alpha_l * img[y1:y2, x1:x2, c])

