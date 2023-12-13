import matplotlib.pyplot as plt # For displaying the figures
import cv2 # opencv
import numpy as np # For numerical calculations
import time # for the calculation of the execution time

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
    image = np.array(image0, dtype=np.uint16)   # transformation of the image into uint16 so that each pixel of the
                                                # image will have the same intensity change (min value = 0, max value = 65535)
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
    # # Save the images to disk
    # for i, img in enumerate(List8Img):
    #     filename = f"{basedir}intensity/image_{i}.png"  # You can change the format and naming convention as needed
    #     cv2.imwrite(filename, img)
    print("Scenario 1 Intensity: 4 images with intensity change I+b and 4 images with intensity change I*c")
    return imageO, List8Img
# ................................................................................

## Scenario 2 (Scale): Function that takes as input the index of the camera, the index of the image n, and a scale, it returns
#                      a couple (I, Iscale). In the following, we will work with 7 images with a scale change Is : s ∈]1.1 : 0.2 : 2.3].
def get_cam_scale(s):
    Img = cv2.imread(data)
    Img = np.array(Img) # transform the image into an array type
    ImgScale = cv2.resize(Img, (0, 0), fx=s, fy=s, interpolation = cv2.INTER_NEAREST) # opencv resize function with INTER_NEAREST interpolation
    I_Is = list([Img, ImgScale]) # list of 2 images (original image and scaled image)
    # # Save the images to disk
    # filename = f"{basedir}scale/image_{s}.png"  # You can change the format and naming convention as needed
    # cv2.imwrite(filename, ImgScale)
    print("get_cam_scale run with parameter s = ", s, " and return a couple (I, Iscale) of images")
    return I_Is
# ................................................................................

## Scenario 3 (Rotation): Function that takes as input the index of the camera, the index of the image n, and a rotation angle, it returns a
#                         couple (I, Irot), and the rotation matrix. In the following, we will work with 9 images with a change of scale For
#                         an image I, we will create 9 images (I10, I20...I90) with change of rotation from 10 to 90 with a step of 10.
def get_cam_rot(r):
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
    # # Save the images to disk
    # filename = f"{basedir}rotation/image_{r}.png"  # You can change the format and naming convention as needed
    # cv2.imwrite(filename, rotated_image)
    print("get_cam_rot run with parameter r = ", r, " and return a couple (I, Irot) of images")
    return rotate_matrix,couple_I_Ir # it also returns the rotation matrix for further use in the rotation evaluation function
# ................................................................................

# ...................................................................................................................
# I.2 Scenario evaluation: Function for each scenario that returns the percentage of the match of two lists of correct matched points
# ...................................................................................................................

## Evaluation of scenario 1: Intensity change: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching, it returns the percentage of correct matched points
def evaluate_scenario_1(KP1, KP2, Dspt1, Dspt2, match_method):
# For this scenario1, the evaluation between two images with change of intensity, we must compare only the coordinates (x,y) of the detected
# points between the two images.

    # creation of a feature matcher
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
    # match the descriptors of the two images
    Dspt1 = Dspt1.astype(np.float32)
    Dspt2 = Dspt2.astype(np.float32)
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

## Evaluation of scenario 2: Scale change: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching and the scale, it returns the percentage of correct matched points
def evaluate_scenario_2(KP1, KP2, Dspt1, Dspt2, match_method,scale):
# For this scenario2, the evaluation between two images with change of scale, we must compare the coordinates (x,y)
# of the detected points between the two images (I and I_scale), after multiplying by the scale the coordinates
# of the detected points in I_scale.

    # creation of a feature matcher
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
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

## Evaluation of scenario 3: Rotation change: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching, the rotation angle and the rotation matrix, it returns the percentage of correct matched points
def evaluate_scenario_3(KP1, KP2, Dspt1, Dspt2, match_method, rot, rot_matrix):
# For this scenario3, the evaluation between two images with rotation change, we must compare the coordinates (x,y)
# of the points detected between the two images (I and I_scale), after multiplying by rot_matrix[:2,:2] the coordinates
# of the points detected in I_rotation by adding a translation rot_matrix[0,2] for x and rot_matrix[1,2] for y.
    
    # ccreation of a feature matcher
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
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

# Initialization of our methods of detectors and descriptors (17 methods)
### detectors/descriptors 5
sift  = cv2.SIFT_create()
akaze = cv2.AKAZE_create()
orb   = cv2.ORB_create()
brisk = cv2.BRISK_create()
kaze  = cv2.KAZE_create()
### detectors 8
fast  = cv2.FastFeatureDetector_create()
star  = cv2.xfeatures2d.StarDetector_create()
mser  = cv2.MSER_create()
agast = cv2.AgastFeatureDetector_create()
gftt  = cv2.GFTTDetector.create()
harrislaplace = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
msd   = cv2.xfeatures2d.MSDDetector_create()
tbmr  = cv2.xfeatures2d.TBMR_create()
### descriptors 9
vgg   = cv2.xfeatures2d.VGG_create()
daisy = cv2.xfeatures2d.DAISY_create()
freak = cv2.xfeatures2d.FREAK_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
lucid = cv2.xfeatures2d.LUCID_create()
latch = cv2.xfeatures2d.LATCH_create()
beblid= cv2.xfeatures2d.BEBLID_create(5.0)
teblid= cv2.xfeatures2d.TEBLID_create(5.0)
boost = cv2.xfeatures2d.BoostDesc_create()

# lists of the different detectors, descriptors and matching methods
DetectDescript = list([sift, akaze, orb, brisk, kaze])
Detectors      = list([fast, star, mser, agast, gftt, harrislaplace, msd, tbmr])
Descriptors    = list([vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost])
matching2      = list([cv2.NORM_L1, cv2.NORM_L2])
# matching3 = list([cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_HAMMING])

################ Scenario 1 (Intensity) ################
print("Scenario 1 Intensity")
Img0 = cv2.imread(data)
Img0 = np.array(Img0)
val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images

## 2 matrices of the rates of scenario 1, the first one gathers the rates for each image, each non-binary method
# (same detectors and descriptors), and each type of matching. And the other one groups the
# rates for each image, each method binary method (different detectors and descriptors), and each type of matching.
Rate_intensity1 = np.zeros((nbre_img, len(matching2), len(DetectDescript)))
Rate_intensity2 = np.zeros((nbre_img, len(matching2), len(Detectors), len(Descriptors)))

img1, HuitImg1 = get_cam_intensity_8Img(Img0, val_b, val_c) # use the intensity change images (I+b and I*c)
# for loop to compute rates (%) for intensity change images, matches, binary and non-binary methods

for k in range(nbre_img): # for the 8 intensity images

    img2 = HuitImg1[k] # image with intensity change
    for c2 in range(len(matching2)): # for bf.L1 and bf.L2 mapping
        matching_method = matching2[c2]
        for ii in range(len(DetectDescript)):
            method = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = method.detectAndCompute(img1, None) # the keypoints and descriptors of the image 1 obtained by the method X
            keypoints22, descriptors22 = method.detectAndCompute(img2, None) # the keypoints and descriptors of the image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 1
            Rate_intensity1[k, c2, ii] = evaluate_scenario_1(keypoints11, keypoints22, descriptors11, descriptors22, matching_method)
    print("Scenario 1 Intensity: Rate_intensity1 for image ", k, " is calculated")
    for c3 in range(len(matching2)): # for bf.L1 and bf.L2 mapping
        match3 = matching2[c3]
        for i in range(len(Detectors)):
            method_keyPt = Detectors[i] # choose a detector from the "Detectors" list
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = method_keyPt.detect(img1,None)
                keypoints2   = method_keyPt.detect(img2,None)
                keypoints1   = method_dscrpt.compute(img1, keypoints1)[0] # the keypoints of image 1 obtained by the method Y
                keypoints2   = method_dscrpt.compute(img2, keypoints2)[0] # the keypoints of image 2 obtained by the method Y
                descriptors1 = method_dscrpt.compute(img1, keypoints1)[1] # the descriptors of the image 1 obtained by the method Y
                descriptors2 = method_dscrpt.compute(img2, keypoints2)[1] # the descriptors of the image 2 obtained by the method Y
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 1
                Rate_intensity2[k, c3, i, j] = evaluate_scenario_1(keypoints1, keypoints2, descriptors1, descriptors2, match3)
    print("Scenario 1 Intensity: Rate_intensity2 for image ", k, " is calculated")

# export numpy arrays
np.save(basedir + 'arrays/Rate_intensity1.npy', Rate_intensity1)
np.save(basedir + 'arrays/Rate_intensity2.npy', Rate_intensity2)
##########################################################

################ Scenario 2: Scale ################
print("Scenario 2 Scale")
scale = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3] # 7 values of the scale change s ∈]1.1 : 0.2 : 2.3].

## 2 matrices of the rates of scenario 2, the first one groups the rates for each image, each non-binary method (same detectors and descriptors),
# and each type of matching. And the other one groups the rates for each image, each binary method (different detectors and
# descriptors), and each type of matching.
Rate_scale1 = np.zeros((len(scale), len(matching2), len(DetectDescript)))
Rate_scale2 = np.zeros((len(scale), len(matching2), len(Detectors), len(Descriptors)))
# for loop to calculate rates (%) for scaling images, matching, binary and non-binary methods
for s in range(len(scale)): # for the 7 scale images
    # use the original image and the scaling image (I and Is)
    img = get_cam_scale(scale[s])#[0] # image I
    #img2 = get_cam_scale(scale[s])[1] # image Is

    for c2 in range(len(matching2)): # for bf.L1 and bf.L2 mapping
        matching_method = matching2[c2]
        for ii in range(len(DetectDescript)):
            method = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = method.detectAndCompute(img[0], None)# the keypoints and descriptors of image 1 obtained by the method X
            keypoints22, descriptors22 = method.detectAndCompute(img[1], None)# the keypoints and descriptors of image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 2
            Rate_scale1[s, c2, ii] = evaluate_scenario_2(keypoints11, keypoints22, descriptors11, descriptors22, matching_method, scale[s])
    print("Scenario 2 Scale: Rate_scale1 for image ", s, " is calculated")
    for c3 in range(len(matching2)): # for bf.L1, bf.L2
        matching_method = matching2[c3]
        for i in range(len(Detectors)):
            method_keyPt = Detectors[i] # choose a detector from the "Detectors" list
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = method_keyPt.detect(img[0],None)
                keypoints2   = method_keyPt.detect(img[1],None)
                keypoints1   = method_dscrpt.compute(img[0], keypoints1)[0] # the keypoints of image 1 obtained by the method Y
                keypoints2   = method_dscrpt.compute(img[1], keypoints2)[0] # the keypoints of image 2 obtained by the method Y
                descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1] # the descriptors of the image 1 obtained by the method Y
                descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1] # the descriptors of the image 2 obtained by the method Y
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 2
                Rate_scale2[s, c3, i, j] = evaluate_scenario_2(keypoints1, keypoints2, descriptors1, descriptors2, matching_method, scale[s])
    print("Scenario 2 Scale: Rate_scale2 for image ", s, " is calculated")
# export numpy arrays
np.save(basedir + 'arrays/Rate_scale1.npy', Rate_scale1)
np.save(basedir + 'arrays/Rate_scale2.npy', Rate_scale2)
##########################################################

################ Scenario 3: Rotation ################
print("Scenario 3 Rotation")
rot = [10, 20, 30, 40, 50, 60, 70, 80, 90] # 9 values of rotation change, rotations from 10 to 90 with a step of 10.

## 2 matrices of the rates of scenario 3, the first one groups the rates for each image, each non-binary method (same detectors and descriptors),
# and each type of matching. And the other one groups the rates for each image, each binary method (different detectors and
# descriptors), and each type of matching.
Rate_rot1 = np.zeros((len(rot), len(matching2), len(DetectDescript)))
Rate_rot2 = np.zeros((len(rot), len(matching2), len(Detectors), len(Descriptors)))
# for loop to compute rates (%) for rotation change images, matches, binary and non-binary methods
for r in range(len(rot)):
    # use the rotation matrix, the original image and the rotation change matrix (I and Ir)
    rot_matrix, img = get_cam_rot(rot[r])
    img1 = img[0] # image I
    img2 = img[1] # image Ir

    for c2 in range(len(matching2)): # for bf.L1 and bf.L2 mappings
        matching_method = matching2[c2]
        for ii in range(len(DetectDescript)):
            method = DetectDescript[ii] # choose a method from the "DetectDescript" list
            keypoints11, descriptors11 = method.detectAndCompute(img1, None)# the keypoints and descriptors of image 1 obtained by the method X
            keypoints22, descriptors22 = method.detectAndCompute(img2, None)# the keypoints and descriptors of image 2 obtained by the method X
            # Calculation of the rate (%) of correctly matched homologous points by the X method using the evaluation function of scenario 3
            Rate_rot1[r, c2, ii] = evaluate_scenario_3(keypoints11, keypoints22, descriptors11, descriptors22, matching_method, rot[r], rot_matrix)
    print("Scenario 3 Rotation: Rate_rot1 for image ", r, " is calculated")
    for c3 in range(len(matching2)): # for bf.L1 and bf.L2
        matching_method = matching2[c3]
        for i in range(len(Detectors)):
            method_keyPt = Detectors[i] # choose a detector from the "Detectors" list
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j] # choose a descriptor from the "Descriptors" list
                keypoints1   = method_keyPt.detect(img1,None)
                keypoints2   = method_keyPt.detect(img2,None)
                keypoints1   = method_dscrpt.compute(img1, keypoints1)[0]# the keypoints of image 1 obtained by the method Y
                keypoints2   = method_dscrpt.compute(img2, keypoints2)[0]# the keypoints of image 2 obtained by the method Y
                descriptors1 = method_dscrpt.compute(img1, keypoints1)[1]# the descriptors of the image 1 obtained by the method Y
                descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]# the descriptors of the image 2 obtained by the method Y
                # Calculation of the rate (%) of correctly matched homologous points by the Y method using the evaluation function of scenario 3
                Rate_rot2[r, c3, i, j] = evaluate_scenario_3(keypoints1, keypoints2, descriptors1, descriptors2, matching_method, rot[r], rot_matrix)
    print("Scenario 3 Rotation: Rate_rot2 for image ", r, " is calculated")
# export numpy arrays
np.save(basedir + 'arrays/Rate_rot1.npy', Rate_rot1)
np.save(basedir + 'arrays/Rate_rot2.npy', Rate_rot2)
##########################################################