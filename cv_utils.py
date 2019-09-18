import cv2
import math
import numpy as np
from PIL import Image
import pytesseract

"""
List of helper, enhancement and transformation funcs commonly used for OCR and
CV processing given image or bounding box information. N.B. bbox format in
Tensorflow Object Detection format, i.e. [y_min, x_min, y_max, x_max]

List of functions:

apply_OCR():       Applies Tesseract OCR library to extract a string from image
auto_rotate_img(): Applies Canny transform and open-cv Hough Line transform
                   to auto rotate image.
crop_bbox():       Crop a bounding box region from an image.
draw_bbox():       Draws a single bounding box on image.
draw_bboxes():     Draws multiple bounding boxes on image.
expand_bbox():     Expand the bounding box passed in by a prespecified factor.
"""


def apply_OCR(img, img_num):
    """
    Applies Tesseract OCR library to extract a string from image.
    Modified code from https://github.com/stefbo/ulr-ocr
    Input: Image
    Output: String
    """
    ret_string = "test"
    SCREEN_WIDTH = 128
    SCREEN_HEIGHT = 64

    if img is None:
        print("Reading image failed, img is None")
        exit(1)

    # Extract the center part from the image. Quality of the OCR
    # for the header line and result unit is poor. The parse the
    # unit using template matching below.

    # Re-scale the image before passing it to tesseract and do
    # automatic thresholding using Otsu's algorithm.
    scale_factor = 4
    scaled_img = cv2.resize(img[0:50, 0:SCREEN_WIDTH], (0, 0), fx=scale_factor,
                            fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # DEBUG: originally cv2.resize(img[10:50, ...])
    assert(scaled_img is not None)

    thres, thres_img = cv2.threshold(scaled_img, 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    assert(thres_img is not None)

    text = pytesseract.image_to_string(
        thres_img, config='--user-words words.txt config.txt')
    ret_string = text

    # TODO: TEMPLATE MATCHING EXTENSION. CHECK ORIGINAL REPO 4 DETAILS
    return ret_string


def auto_rotate_img(img_before):
    """ Applies Canny transform and open-cv Hough Line transform to auto rotate
    image.
    """
    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    # Probabalistic Hough Lines Transform
    # Input: binary img, rho accuracy, theta accuracy, threshold(min no of
    #        votes to be considered a line, so if 2 points in a line, then 2),
    #        minLineLength, maxLineGap
    # Returns: Two endpoints of each line
    lines = cv2.HoughLinesP(img_edges, 0.01, np.pi/180, 10, minLineLength=10,
                            maxLineGap=1)
    if lines is None:
        print("\u001b[31mN.B.: HoughLines detected no lines\u001b[30m")
        return img_gray

    angles = []

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

    median_angle = np.median(angles)
    print("Suggested angle is: ", str(median_angle))
    if abs(median_angle) > 20:
        print("Suggested angle too large. Disregarding.")
        img_rotated = img_gray
    else:
        print("Rotating by suggested angle.")
        img_rotated = ndimage.rotate(img_gray, median_angle)

    return img_rotated


def crop_bbox(image, y_min, x_min, y_max, x_max):
    """
    Crop a bounding box region from an image.
    Input: Image from which to extract a bounding box, bounding box coordinates
    Output: Cropped image using the bounding box
    """
    img = np.asarray(image)  # N.B. np.asarray converts (WxHxC) to (HxWxC)
    img_width = img.shape[1]
    img_height = img.shape[0]

    # Convert bbox coord from ratio to integer relative to the size of the img
    x_min_quant = int(np.floor(x_min*img_width))
    x_max_quant = int(np.ceil(x_max*img_width))
    y_min_quant = int(np.floor(y_min*img_height))
    y_max_quant = int(np.ceil(y_max*img_height))
    img_ret = Image.fromarray(img[y_min_quant: y_max_quant,
                                  x_min_quant: x_max_quant, :], 'RGB')
    return img_ret


def draw_bbox(original_image, y_min, x_min, y_max, x_max, thickness,
              color=(255, 0, 0)):
    """ Draws a single bounding box on image.
    Input: Original image, bbox coordinates, thickness of line and color.
    Output: Image with drawn bbox
    Update note: mod draw_bboxes() accordingly
    """

    # N.B. np.asarray converts (WxHxC) to (HxWxC)
    original_image_np = np.asarray(original_image)
    img_width = original_image_np.shape[1]
    img_height = original_image_np.shape[0]
    return_image = np.copy(original_image_np)
    x_min_px = int(x_min*img_width)
    y_min_px = int(y_min*img_height)
    x_max_px = int(x_max*img_width)
    y_max_px = int(y_max*img_height)
    cv2.rectangle(return_image, (x_min_px, y_min_px), (x_max_px, y_max_px),
                  color, thickness)
    return return_image


def draw_bboxes(original_image, bboxes, num, thickness, color=(255, 0, 0)):
    """ Draws multiple bounding boxes on image
    Input: Original image, array of bboxes each with the format (y_min, x_min,
           y_max, y_max) number of bboxes to draw, thickness of line and color
    Output: Image with drawn bboxes
    Update note: mod draw_bbox() accordingly
    """

    # N.B. np.asarray converts (WxHxC) to (HxWxC)
    original_image_np = np.asarray(original_image)
    img_width = original_image_np.shape[1]
    img_height = original_image_np.shape[0]
    return_image = np.copy(original_image_np)
    if num > len(bboxes):
        print("Error: num argument > length of bboxes")
        return None
    for i in range(num):
        bbox = bboxes[i]
        x_min = bbox[1]
        y_min = bbox[0]
        x_max = bbox[3]
        y_max = bbox[2]
        x_min_px = int(x_min*img_width)
        y_min_px = int(y_min*img_height)
        x_max_px = int(x_max*img_width)
        y_max_px = int(y_max*img_height)
        cv2.rectangle(return_image, (x_min_px, y_min_px), (x_max_px, y_max_px),
                      color, thickness)
    return return_image


def expand_bbox(image, y_min, x_min, y_max, x_max, factor):
    """ Expand the bounding box passed in by a prespecified factor.
    Output: Tuple of (new_x_min, new_y_min, new_x_max, new_y_max)
    """

    # N.B. np.asarray converts (WxHxC) to (HxWxC)
    np_image = np.asarray(image)
    img_width = np_image.shape[1]
    img_height = np_image.shape[0]
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0
    x_length = (x_max - x_min)
    y_length = (y_max - y_min)
    new_x_length = x_length * factor
    new_y_length = y_length * factor
    new_x_min = x_center - (new_x_length/2.0)
    new_y_min = y_center - (new_y_length/2.0)
    new_x_max = x_center + (new_x_length/2.0)
    new_y_max = y_center + (new_y_length/2.0)
    if factor <= 0:
        print("Error: factor must be positive")
        return None
    if (new_x_min < 0 or new_y_min < 0 or new_x_max > img_width
            or new_y_max > img_height):
        print("Error: cannot expand since new bbox is out of bounds")
        # TODO: Need to handle the fact that the len of bbox is now wrong
        return [max(0, new_y_min), max(0, new_x_min),
                min(img_height, new_y_max), min(img_width, new_x_max)]
    return [new_y_min, new_x_min, new_y_max, new_x_max]
