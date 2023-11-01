##    Comparative study of detectors, descriptors and matching    ##
##                     of points of interest                      ##

## Objective: In this project, we tested and evaluated the performance of several most used interest point detectors
# such as (SIFT, AKAZE, ORB, BRISK, KAZE, FAST, STAR and MSER), several descriptors such as (SIFT, AKAZE, ORB, BRISK,
# KAZE, FREAK, LATCH, LUCID and BRIEF) and several matching methods such as (Brute-Force L1, Brute-Force L2 and
# Brute-Force HAMMING), in order to know the most suitable method for a given scenario.

# ................................................................................
## Imports of libraries
import matplotlib.pyplot as plt # For displaying the figures
import cv2 # opencv
import numpy as np # For numerical calculations
import time # for the calculation of the execution time
from prettytable import PrettyTable # To view the displayboards on the console

## Reading database
# basedir = '/content/drive/MyDrive/Dataset/oxfordAffine'
basedir = './'
folder = '/bikes'
picture = '/img1.jpg'
data = basedir + folder + picture

# ...................................................................................................................
# I.1 Data preparation
# ...................................................................................................................

## Scenario 1 (Intensity): Function that returns 8 images with intensity changes from an I image.
def get_cam_intensity_8Img(image0, val_b, val_c): # val_b, val_c must be 2 verctors with 4 values each
    imageO = np.array(image0)
    image = np.array(image0, dtype=np.uint16) # transformation of the image into uint16 so that each pixel of the
#                                               image will have the same intensity change (min value = 0, max value = 65535)
    I0 = np.zeros((image.shape[0], image.shape[1], image.shape[2])) # creation of empty image of 3 chanels to fill it afterwards
    List8Img = list([I0, I0, I0, I0, I0, I0, I0, I0]) # list of our 8 images that we will create
    for i in range(len(val_b)): # for I + b, with: b ∈ [-30 : 20 : +30]
        I =  image + val_b[i]
        List8Img[i] = I.astype(int)
        List8Img[i][List8Img[i] > 255] = 255 # set pixels with intensity > 255 to 255
        List8Img[i][List8Img[i] < 0] = 0 # set the pixels with intensity < 0 to the value of 0
        List8Img[i] = np.array(List8Img[i], dtype=np.uint8) # image transformation to uint8
    for j in range(len(val_c)): # for I ∗ c, with: c ∈ [0.7 : 0.2 : 1.3].
        I =  image * val_c[j]
        List8Img[j+4] = I.astype(int)
        List8Img[j+4][List8Img[j+4] > 255] = 255 # set pixels with intensity > 255 to 255
        List8Img[j+4][List8Img[j+4] < 0] = 0 # set the pixels with intensity < 0 to the value of 0
        List8Img[j+4] = np.array(List8Img[j+4], dtype=np.uint8) # transform image to uint8 (min value = 0, max value = 255)
    return imageO, List8Img
# ................................................................................

## Scenario 2 (Scale): Function that takes as input the index of the camera, the index of the image n, and a scale, it returns
#                      a couple (I, Iscale). In the following, we will work with 7 images with a scale change Is : s ∈]1.1 : 0.2 : 2.3].
def get_cam_scale(camN, n, s):
    Img = cv2.imread(data)
    Img = np.array(Img) # transform the image into an array type
    ImgScale = cv2.resize(Img, (0, 0), fx=s, fy=s, interpolation = cv2.INTER_NEAREST) # opencv resize function with INTER_NEAREST interpolation
    I_Is = list([Img, ImgScale]) # list of 2 images (original image and scaled image)
    return I_Is
# ................................................................................

## Scenario 3 (Rotation): Function that takes as input the index of the camera, the index of the image n, and a rotation angle, it returns a
#                         couple (I, Irot), and the rotation matrix. In the following, we will work with 9 images with a change of scale For
#                         an image I, we will create 9 images (I10, I20...I90) with change of rotation from 10 to 90 with a step of 10.
def get_cam_rot(camN, n, r):
    Img = cv2.imread(data)
    Img = np.array(Img) # transform the image into an array type
    # divide the height and width by 2 to get the center of the image
    height, width = Img.shape[:2]
    # get the coordinates of the center of the image to create the 2D rotation matrix
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=r, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(Img, rotate_matrix, dsize=(width, height), flags=cv2.INTER_LINEAR)
    couple_I_Ir = list([Img, rotated_image]) # list of 2 images (original image and image with rotation change)
    return rotate_matrix,couple_I_Ir # it also returns the rotation matrix for further use in the rotation evaluation function
# ................................................................................

# ...................................................................................................................
# I.2 Scenario evaluation: Function for each scenario that returns the percentage of the match of two lists of correct matched points
# ...................................................................................................................

## Evaluation of scenario 1: Function that takes as input the keypoints, the descriptors (of 2 images) and the type of matching, it returns
#                            the percentage of correct matched points
def evaluate_scenario_1(KP1, KP2, Dspt1, Dspt2, mise_corresp):
# For this scenario1, the evaluation between two images with change of intensity, we must compare only the coordinates (x,y) of the detected
# points between the two images.

    # creation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True)
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)

    Prob_P = 0
    Prob_N = 0

    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])

        # comparison between these coordinates (x,y)
        if (abs(X1 - X2) <=2) and (abs(Y1 - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1
        else:
            Prob_N += 1
    # Calculation of the rate (%) of correctly matched homologous points
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100

    return Prob_True
# ................................................................................

## Evaluation of scenario 2: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching and the scale, it returns the percentage of correct matched points
def evaluate_scenario_2(KP1, KP2, Dspt1, Dspt2, mise_corresp,scale):
# For this scenario2, the evaluation between two images with change of scale, we must compare the coordinates (x,y)
# of the detected points between the two images (I and I_scale), after multiplying by the scale the coordinates
# of the detected points in I_scale.

    # creation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True)
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)

    Prob_P = 0
    Prob_N = 0

    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])

        if (abs(X1*scale - X2) <=2) and (abs(Y1*scale - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1
        else:
            Prob_N += 1
    # Calculation of the rate (%) of correctly matched homologous points
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100

    return Prob_True
# ................................................................................

## Evaluation of scenario 3: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching, the degree of rotation and the rotation matrix, it returns
#                            the percentage of correct matched points
def evaluate_scenario_3(KP1, KP2, Dspt1, Dspt2, mise_corresp,rot, rot_matrix):
# For this scenario3, the evaluation between two images with rotation change, we must compare the coordinates (x,y)
# of the points detected between the two images (I and I_scale), after multiplying by rot_matrix[:2,:2] the coordinates
# of the points detected in I_rotation by adding a translation rot_matrix[0,2] for x and rot_matrix[1,2] for y.
    
    # ccreation of a feature matcher
    bf = cv2.BFMatcher(mise_corresp, crossCheck=True)
    # match the descriptors of the two images
    matches = bf.match(Dspt1,Dspt2)
    # Sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)

    Prob_P = 0
    Prob_N = 0
    theta = rot*(np.pi/180) # transformation of the degree of rotation into radian
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])

        X12 = X1*np.cos(theta) + Y1*np.sin(theta) + rot_matrix[0,2]
        Y12 = -X1*np.sin(theta) + Y1*np.cos(theta) + rot_matrix[1,2]

        if (abs(X12 - X2) <=2) and (abs(Y12 - Y2) <=2):   #  Tolerance allowance (∼ 1-2 pixels)
            Prob_P += 1
        else:
            Prob_N += 1
    # Calculation of the rate (%) of correctly matched homologous points
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100

    return Prob_True
# ................................................................................

# ...................................................................................................................
# II. Evaluation of matching methods for scenarios 1, 2 and 3
# ...................................................................................................................

# Initialization of our methods of detectors and descriptors (17 methods)
### detectors/descriptors
sift  = cv2.SIFT_create()
akaze = cv2.AKAZE_create()
orb   = cv2.ORB_create()
brisk = cv2.BRISK_create()
kaze  = cv2.KAZE_create()
### detectors
fast  = cv2.FastFeatureDetector_create()
star  = cv2.xfeatures2d.StarDetector_create()
mser  = cv2.MSER_create()
agast = cv2.AgastFeatureDetector_create()
gftt  = cv2.GFTTDetector.create()
harrislaplace = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
msd   = cv2.xfeatures2d.MSDDetector_create()
tbmr  = cv2.xfeatures2d.TBMR_create()
### descriptors
freak = cv2.xfeatures2d.FREAK_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
lucid = cv2.xfeatures2d.LUCID_create()
latch = cv2.xfeatures2d.LATCH_create()
beblid= cv2.xfeatures2d.BEBLID_create(5.0)
teblid= cv2.xfeatures2d.TEBLID_create(5.0)
boost = cv2.xfeatures2d.BoostDesc_create()
vgg   = cv2.xfeatures2d.VGG_create()
daisy = cv2.xfeatures2d.DAISY_create()

# lists of the different detectors, descriptors and matching methods
DetectDescript = list([sift, akaze, orb, brisk, kaze])
Detectors     = list([fast, star, mser, agast, gftt, harrislaplace, msd, tbmr])
Descriptors   = list([vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost])
matching2 = list([cv2.NORM_L1, cv2.NORM_L2])
matching3 = list([cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_HAMMING])
# ................................................................................


################ Scenario 1 (Intensity)
print("Scenario 1 Scale")
scenario1_time = time.time()
#Img0 = data.get_cam2(0) # Original image
Img0 = cv2.imread(data)
Img0 = np.array(Img0)
val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images

## 2 matrices of the rates of scenario 1, the first one gathers the rates for each image, each non-binary method
# (same detectors and descriptors), and each type of matching (without bf.HAMMING). And the other one groups the
# rates for each image, each method binary method (different detectors and descriptors), and each type of matching (with bf.HAMMING).
Rate_intensity1 = np.zeros((nbre_img, len(matching2), len(DetectDescript)))
Rate_intensity2 = np.zeros((nbre_img, len(matching3), len(Detectors), len(Descriptors)))

img1, HuitImg1 = get_cam_intensity_8Img(Img0, val_b, val_c) # use the intensity change images (I+b and I*c)
# for loop to compute rates (%) for intensity change images, matches, binary and non-binary methods

for k in range(nbre_img): # for the 8 intensity images
    # Start the timer
    start_time = time.time()

    img2 = HuitImg1[k] # image with intensity change
    for c2 in range(len(matching2)): # for bf.L1 and bf.L2 mapping (bf.HAMMING does not work for most non-binary methods)
        match = matching2[c2]
        for ii in range(len(DetectDescript)):
            method = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = method.detectAndCompute(img1, None) # the keypoints and descriptors of the image 1 obtained by the method X
            keypoints22, descriptors22 = method.detectAndCompute(img2, None) # the keypoints and descriptors of the image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 1
            Rate_intensity1[k, c2, ii] = evaluate_scenario_1(keypoints11, keypoints22, descriptors11, descriptors22, match)

    elapsed_time = time.time() - start_time
    print(f"SCenario 1 - c2 Elapsed time: {elapsed_time} seconds on image {k}")

    start_time = time.time()
    for c3 in range(len(matching3)): # for bf.L1, bf.L2 and bf.HAMMING mapping
        match = matching3[c3]
        for i in range(len(Detectors)):
            method_keyPt = Detectors[i] # choose a detector from the "Detectors" list
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = method_keyPt.detect(img1,None)
                keypoints2   = method_keyPt.detect(img2,None)
                keypoints1   = method_dscrpt.compute(img1, keypoints1)[0] # the keypoints of image 1 obtained by the method Y
                keypoints2   = method_dscrpt.compute(img2, keypoints2)[0] # the keypoints of image 2 obtained by the method Y
                descriptors1 = method_dscrpt.compute(img1, keypoints1)[1] # the descriptors of the image 1 obtained by the method Y
                if descriptors1.dtype != np.float32:
                    if descriptors1.dtype != np.uint8:
                        descriptors1 = descriptors1.astype(np.float32)
                descriptors2 = method_dscrpt.compute(img2, keypoints2)[1] # the descriptors of the image 2 obtained by the method Y
                if descriptors2.dtype != np.float32:
                    if descriptors2.dtype != np.uint8:
                        descriptors2 = descriptors2.astype(np.float32)
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 1
                Rate_intensity2[k, c3, i, j] = evaluate_scenario_1(keypoints1, keypoints2, descriptors1, descriptors2, match)

    elapsed_time = time.time() - start_time
    print(f"SCenario 1 - c3 Elapsed time: {elapsed_time} seconds on image {k}")

print(f"Scenario 1 Elapsed time: {time.time() - scenario1_time} seconds")
# ................................................................................

################ Scenario 2: Scale
print("Scenario 2 Scale")
scenario2_time = time.time()
cameraN = 2 # camera index
ImageN = 0 # image index
scale = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3] # 7 values of the scale change s ∈]1.1 : 0.2 : 2.3].

## 2 matrices of the rates of scenario 2, the first one groups the rates for each image, each non-binary method (same detectors and descriptors),
# and each type of matching (without bf.HAMMING). And the other one groups the rates for each image, each binary method (different detectors and
# descriptors), and each type of matching (with bf.HAMMING).
Rate_scale1 = np.zeros((len(scale), len(matching2), len(DetectDescript)))
Rate_scale2 = np.zeros((len(scale), len(matching3), len(Detectors), len(Descriptors)))
# for loop to calculate rates (%) for scaling images, matching, binary and non-binary methods
for s in range(len(scale)): # for the 7 scale images
    # use the original image and the scaling image (I and Is)
    img1 = get_cam_scale(cameraN, ImageN, scale[s])[0] # image I
    img2 = get_cam_scale(cameraN, ImageN, scale[s])[1] # image Is

    start_time = time.time()

    for c2 in range(len(matching2)): # for bf.L1 and bf.L2 mapping (bf.HAMMING does not work for most non-binary methods)
        match = matching2[c2]
        for ii in range(len(DetectDescript)):
            method = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = method.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by the method X
            keypoints22, descriptors22 = method.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 2
            Rate_scale1[s, c2, ii] = evaluate_scenario_2(keypoints11, keypoints22, descriptors11, descriptors22, match, scale[s])

    elapsed_time = time.time() - start_time
    print(f"SCenario 2 - c2 Elapsed time: {elapsed_time} seconds on image {s}")
    start_time = time.time()

    for c3 in range(len(matching3)): # for bf.L1, bf.L2 and bf.HAMMING mapping
        match = matching3[c3]
        for i in range(len(Detectors)):
            method_keyPt = Detectors[i] # choose a detector from the "Detectors" list
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = method_keyPt.detect(img1,None)
                keypoints2   = method_keyPt.detect(img2,None)
                keypoints1   = method_dscrpt.compute(img1, keypoints1)[0] # the keypoints of image 1 obtained by the method Y
                keypoints2   = method_dscrpt.compute(img2, keypoints2)[0] # the keypoints of image 2 obtained by the method Y
                descriptors1 = method_dscrpt.compute(img1, keypoints1)[1] # the descriptors of the image 1 obtained by the method Y
                if descriptors1.dtype != np.float32:
                    if descriptors1.dtype != np.uint8:
                        descriptors1 = descriptors1.astype(np.float32)
                descriptors2 = method_dscrpt.compute(img2, keypoints2)[1] # the descriptors of the image 2 obtained by the method Y
                if descriptors2.dtype != np.float32:
                    if descriptors2.dtype != np.uint8:
                        descriptors2 = descriptors2.astype(np.float32)
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 2
                Rate_scale2[s, c3, i, j] = evaluate_scenario_2(keypoints1, keypoints2, descriptors1, descriptors2, match, scale[s])

    elapsed_time = time.time() - start_time
    print(f"SCenario 2 - c3 Elapsed time: {elapsed_time} seconds on image {s}")

print(f"Scenario 2 Elapsed time: {time.time() - scenario2_time} seconds")
# ................................................................................

################ Scenario 3: Rotation
print("Scenario 3 Scale")
scenario3_time = time.time()
cameraN = 2 # camera index
ImageN = 0 # image index
rot = [10, 20, 30, 40, 50, 60, 70, 80, 90] # 9 values of rotation change, rotations from 10 to 90 with a step of 10.

## 2 matrices of the rates of scenario 3, the first one groups the rates for each image, each non-binary method (same detectors and descriptors),
# and each type of matching (without bf.HAMMING). And the other one groups the rates for each image, each binary method (different detectors and
# descriptors), and each type of matching (with bf.HAMMING).
Rate_rot1 = np.zeros((len(rot), len(matching2), len(DetectDescript)))
Rate_rot2 = np.zeros((len(rot), len(matching3), len(Detectors), len(Descriptors)))
# for loop to compute rates (%) for rotation change images, matches, binary and non-binary methods
for r in range(len(rot)):
    # use the rotation matrix, the original image and the rotation change matrix (I and Ir)
    rot_matrix, img = get_cam_rot(cameraN, ImageN, rot[r])
    img1 = img[0] # image I
    img2 = img[1] # image Ir

    start_time = time.time()

    for c2 in range(len(matching2)): # for bf.L1 and bf.L2 mappings (bf.HAMMING does not work for most non-binary methods)
        match = matching2[c2]
        for ii in range(len(DetectDescript)):
            method = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = method.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by the method X
            keypoints22, descriptors22 = method.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 3
            Rate_rot1[r, c2, ii] = evaluate_scenario_3(keypoints11, keypoints22, descriptors11, descriptors22, match, rot[r], rot_matrix)

    elapsed_time = time.time() - start_time
    print(f"SCenario 3 - c2 Elapsed time: {elapsed_time} seconds on image {r}")

    start_time = time.time()

    for c3 in range(len(matching3)): # for bf.L1, bf.L2 and bf.HAMMING mapping
        match = matching3[c3]
        for i in range(len(Detectors)):
            method_keyPt = Detectors[i] # choose a detector from the "Detectors" list
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = method_keyPt.detect(img1,None)
                keypoints2   = method_keyPt.detect(img2,None)
                keypoints1   = method_dscrpt.compute(img1, keypoints1)[0]# the keypoints of image 1 obtained by the method Y
                keypoints2   = method_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the method Y
                descriptors1 = method_dscrpt.compute(img1, keypoints1)[1]# the descriptors of the image 1 obtained by the method Y
                if descriptors1.dtype != np.float32:
                    if descriptors1.dtype != np.uint8:
                        descriptors1 = descriptors1.astype(np.float32)
                descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]# the descriptors of the image 2 obtained by the method Y
                if descriptors2.dtype != np.float32:
                    if descriptors2.dtype != np.uint8:
                        descriptors2 = descriptors2.astype(np.float32)
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 3
                Rate_rot2[r, c3, i, j] = evaluate_scenario_3(keypoints1, keypoints2, descriptors1, descriptors2, match, rot[r], rot_matrix)

    elapsed_time = time.time() - start_time
    print(f"SCenario 3 - c3 Elapsed time: {elapsed_time} seconds on image {r}")

print(f"Scenario 3 Elapsed time: {time.time() - scenario3_time} seconds")

# ..........................................................................................................................
# Visualization of the results
DetectDescript = list([sift, akaze, orb, brisk, kaze])
Detectors     = list([fast, star, mser, agast, gftt, harrislaplace, msd, tbmr])
Descriptors   = list([vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost])
# ..........................................................................................................................

# Binary and non-binary methods used to set the legend
DetectDescriptLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze']
DetectorsLegend     = ['fast-', 'star-', 'mser-', 'agast-', 'gftt-', 'harrislaplace-', 'msd-', 'tbmr-']
DescriptorsLegend   = ['vgg', 'daisy', 'freak', 'brief', 'lucid', 'latch', 'beblid', 'teblid', 'boost']

c2 = 1 # for non-binary methods "DetectDescript" (c2=0 for bf.L1, c2=1 for bf.L2)
c3 = 2 # for binary methods "Detectors with Descriptors" (c2=0 for bf.L1, c2=1 for bf.L2, c2=2 for bf.HAMMING)
# To choose the type of mapping for our binary and non-binary methods (this is for a good visualization, to
# avoid plotting 46 curves and plotting only 17 curves in each figure)

# Number of colors to use for all curves
NUM_COLORS = len(DetectDescriptLegend) + (len(DetectorsLegend)*len(DescriptorsLegend)) # NUM_COLORS = 17

LINE_STYLES = ['solid', 'dashed', 'dotted'] # style of the curve
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('gist_rainbow')
num = -1 # for plot
# Initialization of the 4 figures
fig1 = plt.figure(1,figsize= (15,10))
fig2 = plt.figure(2,figsize= (15,10))
fig3 = plt.figure(3,figsize= (15,10))
fig4 = plt.figure(4,figsize= (15,10))
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

# for the plot, I have inserted the following link: https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
# for loop to display the results of non-binary methods
for k in range(len(DetectDescriptLegend)):
    Rate1_I1 = Rate_intensity1[:4, c2, k]
    Rate1_I2 = Rate_intensity1[4:, c2, k]
    Rate1_S = Rate_scale1[:, c2, k]
    Rate1_R = Rate_rot1[:, c2, k]

    lines_I1 = ax1.plot(val_b, Rate1_I1, linewidth=2, label = DetectDescriptLegend[k]) # for the figure of the intensity change results (I+b)
    lines_I2 = ax2.plot(val_c, Rate1_I2, linewidth=2, label = DetectDescriptLegend[k]) # for the figure of intensity change results (I*c)
    lines_S = ax3.plot(scale, Rate1_S, linewidth=2, label = DetectDescriptLegend[k]) # for the scaling results figure
    lines_R = ax4.plot(rot, Rate1_R, linewidth=2, label = DetectDescriptLegend[k]) # for the figure of the results of rotation change

    num += 1 # to take each time the loop turns a different color and curve style
    # for the color and style of the curve for the results of the 3 scenarios
    lines_I1[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_I1[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_I2[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_I2[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_S[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_S[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_R[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_R[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])

# for loop to display the results of binary methods
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        Rate2_I1 = Rate_intensity2[:4,c3,i,j]
        Rate2_I2 = Rate_intensity2[4:,c3,i,j]
        Rate2_S = Rate_scale2[:,c3,i,j]
        Rate2_R = Rate_rot2[:,c3,i,j]

        lines_I1 = ax1.plot(val_b, Rate2_I1, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of intensity change results (I+b)
        lines_I2 = ax2.plot(val_c, Rate2_I2, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of intensity change results (I*c)
        lines_S = ax3.plot(scale, Rate2_S, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of the results of scale change
        lines_R = ax4.plot(rot, Rate2_R, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of the results of rotation change

        num += 1 # to take each time the loop turns a different style of curve
        # for the color and style of curve for the results of the 3 scenarios
        lines_I1[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_I1[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
        lines_I2[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_I2[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
        lines_S[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_S[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
        lines_R[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
        lines_R[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])

# The titles of the figures according to the correspondences
if c2 == 0 and c3 == 0:
    ax1.set_title('Results of scenario 1, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
    ax2.set_title('Results of scenario 1, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
    ax3.set_title('Results of scenario 3, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
    ax4.set_title('Results of scenario 4, with bf.L1 for non-binary methods and bf.L1 for binary methods', fontsize=13)
elif c2 == 1 and c3 == 1:
    ax1.set_title('Results of scenario 1, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
    ax2.set_title('Results of scenario 2, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
    ax3.set_title('Results of scenario 3, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
    ax4.set_title('Results of scenario 4, with bf.L2 for non-binary methods and bf.L2 for binary methods', fontsize=13)
elif c2 == 1 and c3 == 2:
    ax1.set_title('Results of scenario 1, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)
    ax2.set_title('Results of scenario 2, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)
    ax3.set_title('Results of scenario 3, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)
    ax4.set_title('Results of scenario 4, with bf.L2 for non-binary methods and bf.HAMMING for binary methods', fontsize=13)

ax1.set_xlabel('Intensity changing (Img +/- value)', fontsize=12) # x-axis title of the figure
ax1.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax1.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 8, handlelength = 2) # legend :(loc=2 <=> Location String = 'upper left')

# ax2.set_title('Correctly matched point rate for different matching methods depending on intensity change', fontsize=13)
ax2.set_xlabel('Intensity changing (Img * value)', fontsize=12) # x-axis title of the figure
ax2.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax2.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 8, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# ax3.set_title('Correctly matched point rate for different matching methods depending on scale change', fontsize=13)
ax3.set_xlabel('Scale changing', fontsize=12) # x-axis title of the figure
ax3.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax3.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 8, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# ax4.set_title('Correctly matched point rate for different pairing methods depending on the change of rotation', fontsize=13)
ax4.set_xlabel('Rotation changing', fontsize=12) # x-axis title of the figure
ax4.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax4.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize= 8, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# Recording and display of the obtained figures
fig1.savefig(basedir + '/figs' + '/Intensity1_changing.png')
fig2.savefig(basedir + '/figs' + '/Intensity1_changing2.png')
fig3.savefig(basedir + '/figs' + '/Scale_changing.png')
fig4.savefig(basedir + '/figs' + '/Rotation_changing.png')
# plt.show()
