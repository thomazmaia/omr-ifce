import cv2
import numpy as np

def disp_image(name, img, time=0):
    cv2.imshow(name, img)
    cv2.waitKey(time)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (a4_width, a4_height), interpolation=cv2.INTER_AREA)
    return img

def pre_process_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    #invert = cv2.bitwise_not(img_binary)
    #kernel = np.ones((3, 3), np.uint8)
    #img_dilation = cv2.dilate(invert, kernel, iterations=1)
    #invert = cv2.bitwise_not(img_dilation)
    return img_binary

def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    return cnts

def get_main_rect(img):
    contours = get_contours(img)

    # height, width = img.shape
    # blank = np.zeros((height, width, 3), np.uint8)
    # for cnt in contours:
    #     cv2.drawContours(blank, cnt, -1, (255,0,0), 2) 
    #     disp_image("blank", blank, 0)

    corners, _ = get_corners(contours[1])
    points = order_points(corners)
    new_img = get_destination_points(points, img)
    return new_img

def get_image_anchors(img):
    contours = get_anchors(img)
    center_points = get_rect_center(contours)
    points = order_points(center_points)
    new_img = get_destination_points(points, img)
    return new_img    

    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 1000
    # params.filterByConvexity = True
    # detector = cv2.SimpleBlobDetector_create(params)    
    # keypoints = detector.detect(img)
    # center_points = order_points(cv2.KeyPoint_convert(keypoints))    
    #new_img = get_destination_points(center_points, img)
    
    return new_img

def get_anchors(img):    
    my_anchors = []            
    height, width = img.shape
    contours = get_contours(img)
    a_ref = cv2.contourArea(contours[0])
    p_ref = cv2.arcLength(contours[0], True)    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if (area >= 0.0045 * a_ref) and (area < 0.006 * a_ref):
            if (perimeter >= 0.035 * p_ref) and (perimeter < 0.055 * p_ref):
                my_anchors.append(cnt)
    return my_anchors

def get_corners(contours):
    img = 255 * np.ones((a4_height,a4_width,1), dtype = np.uint8)
    cv2.drawContours(img, contours, -1, (0,0,0), -1)    
    my_corners = []
    corners = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
    corners = np.intp(corners)
    for i in corners:
        x,y = i.ravel()
        my_corners.append([x,y])
    return my_corners, img

def get_rect_center(contours):
    my_centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        my_centers.append([cX, cY])
    return my_centers

def get_destination_points(corners, image):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights
    -From https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    Args:
        corners: list
    Returns:
        destination_corners: list
        height: int
        width: int
    """
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")  
     
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(points):
    """
    -From: https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    Args:
        img: np.array
        src: list
        dst: list
    Returns:
        un_warped: np.array
    """    
    rect = np.zeros((4, 2), dtype = "float32")
    pts = np.array(points)    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect



path = 'IMG_0861.png'
a4_width = 1240
a4_height = 1754

img = load_image(path)
disp_image("Original", img, 0)

img_process = pre_process_img(img)
disp_image("Processed image", img_process, 0)

img_cropped = get_main_rect(img_process)
disp_image("Main rectangle", img_cropped, 0)

img_anchors = get_image_anchors(img_cropped)
disp_image("All answers", img_anchors, 0)