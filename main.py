import cv2 # opencv
import numpy as np # For numerical calculations

basedir = './oxfordAffine'
folder = '/graf'
picture = '/img1.jpg'
data = basedir + folder + picture

Image = cv2.imread(data)
Image = np.array(Image)

## Scenario 1 (Intensity): Function that returns 8 images with intensity changes from an I image.
def get_intensity_8Img(Img, val_b, val_c): # val_b, val_c must be 2 vectors with 4 values each
    image = np.array(Img, dtype=np.uint16)   # transformation of the image into uint16 so that each pixel of the
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
    # Save the images to disk
    for i, img in enumerate(List8Img):
        filename = f"{basedir}intensity/image_{i}.png"  # You can change the format and naming convention as needed
        cv2.imwrite(filename, img)
    return Img, List8Img
# ................................................................................

## Scenario 2 (Scale): Function that takes as input the index of the camera, the index of the image n, and a scale, it returns
#                      a couple (I, Iscale). In the following, we will work with 7 images with a scale change Is : s ∈]1.1 : 0.2 : 2.3].
def get_cam_scale(Img, s):
    ImgScale = cv2.resize(Img, (0, 0), fx=s, fy=s, interpolation = cv2.INTER_NEAREST) # opencv resize function with INTER_NEAREST interpolation
    I_Is = list([Img, ImgScale]) # list of 2 images (original image and scaled image)
    # Save the images to disk
    filename = f"{basedir}scale/image_{s}.png"
    cv2.imwrite(filename, ImgScale)
    return I_Is
# ................................................................................

## Scenario 3 (Rotation): Function that takes as input the index of the camera, the index of the image n, and a rotation angle, it returns a
#                         couple (I, Irot), and the rotation matrix. In the following, we will work with 9 images with a change of scale For
#                         an image I, we will create 9 images (I10, I20...I90) with change of rotation from 10 to 90 with a step of 10.
def get_cam_rot(Img, r):
    # Get the height and width of the image
    height, width = Img.shape[:2]

    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, r, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_image = cv2.warpAffine(Img, rotation_mat, (bound_w, bound_h))

    couple_I_Ir = [Img, rotated_image]  # list of 2 images (original image and image with rotation change)

    # Save the images to disk
    filename = f"{basedir}rotation/image_{r}.png"  # You can change the format and naming convention as needed
    cv2.imwrite(filename, rotated_image)

    return rotation_mat, couple_I_Ir  # it also returns the rotation matrix for further use in the rotation evaluation function
# ................................................................................

## Evaluation of scenario 1: Intensity change: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching, it returns the percentage of correct matched points
def evaluate_scenario_1(KP1, KP2, Dspt1, Dspt2, match_method):
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
    matches = bf.match(Dspt1,Dspt2)
    matches = sorted(matches, key = lambda x:x.distance)
    Prob_P = 0
    Prob_N = 1
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        if m1 >= len(KP1) or m2 >= len(KP2):
            continue
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])
        # comparison between these coordinates (x,y)
        if (abs(X1 - X2) <=5) and (abs(Y1 - Y2) <=5):   #  Tolerance allowance (∼ 1-2 pixels)
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
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
    matches = bf.match(Dspt1,Dspt2)
    matches = sorted(matches, key = lambda x:x.distance)
    Prob_P = 0
    Prob_N = 1
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        if m1 >= len(KP1) or m2 >= len(KP2):
            continue
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])
        if (abs(X1*scale - X2) <=5) and (abs(Y1*scale - Y2) <=5):   #  Tolerance allowance 5 pixels
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
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
    matches = bf.match(Dspt1,Dspt2)
    matches = sorted(matches, key = lambda x:x.distance)
    Prob_P = 0
    Prob_N = 1
    theta = rot*(np.pi/180) # transformation of the degree of rotation into radian
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        if m1 >= len(KP1) or m2 >= len(KP2):
            continue
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])
        X12 = X1*np.cos(theta) + Y1*np.sin(theta) + rot_matrix[0,2]
        Y12 = -X1*np.sin(theta) + Y1*np.cos(theta) + rot_matrix[1,2]
        if (abs(X12 - X2) <=5) and (abs(Y12 - Y2) <=5):   #  Tolerance allowance 5 pixels
            Prob_P += 1
        else:
            Prob_N += 1
    # Calculation of the rate (%) of correctly matched homologous points
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100
    return Prob_True
# ................................................................................

## Evaluation of scenario 4: graf: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching, it returns the percentage of correct matched points
def evaluate_scenario_4(KP1, KP2, Dspt1, Dspt2, match_method):
    bf = cv2.BFMatcher(normType=match_method, crossCheck=True)
    matches = bf.match(Dspt1,Dspt2)
    matches = sorted(matches, key = lambda x:x.distance)
    Prob_P = 0
    Prob_N = 1
    # A comparison between the coordinates (x,y) of the detected points between the two images => correct and not correct homologous points
    for i in range(len(matches)):
        m1 = matches[i].queryIdx
        m2 = matches[i].trainIdx
        if m1 >= len(KP1) or m2 >= len(KP2):
            continue
        # the coordinates (x,y) of the points detected in the image 1
        X1 = int(KP1[m1].pt[0])
        Y1 = int(KP1[m1].pt[1])
        # the coordinates (x,y) of the points detected in the image 2
        X2 = int(KP2[m2].pt[0])
        Y2 = int(KP2[m2].pt[1])
        if (abs(X1 - X2) <=5) and (abs(Y1 - Y2) <=5):   #  Tolerance allowance 5 pixels
            Prob_P += 1
        else:
            Prob_N += 1
    # Calculation of the rate (%) of correctly matched homologous points
    Prob_True = (Prob_P / (Prob_P + Prob_N))*100
    return Prob_True

### detectors/descriptors 5
sift  = cv2.SIFT.create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=100.0, sigma=1.6)
akaze = cv2.AKAZE.create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE, descriptor_size=0, descriptor_channels=3, threshold=0.00005, nOctaves=4, nOctaveLayers=3, diffusivity=cv2.KAZE_DIFF_PM_G1)
orb   = cv2.ORB.create(nfeatures=500, scaleFactor=1.2, nlevels=4, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=12)
brisk = cv2.BRISK.create(thresh=50, octaves=1, patternScale=1.2)
kaze  = cv2.KAZE.create(extended=False, upright=False, threshold=0.00005,  nOctaves=4, nOctaveLayers=3, diffusivity=cv2.KAZE_DIFF_PM_G2)
### detectors 8
fast  = cv2.FastFeatureDetector.create(threshold=4, nonmaxSuppression=True, type=cv2.FastFeatureDetector_TYPE_9_16)
mser  = cv2.MSER.create(delta=1, min_area=30, max_area=1440, max_variation=0.025, min_diversity=0.8, max_evolution=200, area_threshold=1.01, min_margin=0.003, edge_blur_size=3)
agast = cv2.AgastFeatureDetector.create(threshold=5,nonmaxSuppression=True,type=cv2.AgastFeatureDetector_OAST_9_16)
gftt  = cv2.GFTTDetector.create(maxCorners=20000, qualityLevel=0.002, minDistance=1.0, blockSize=3, useHarrisDetector=False, k=0.04)
star  = cv2.xfeatures2d.StarDetector.create(maxSize=15, responseThreshold=1, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=3)
hl    = cv2.xfeatures2d.HarrisLaplaceFeatureDetector.create(numOctaves=6, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=20000, num_layers=4)
msd   = cv2.xfeatures2d.MSDDetector.create(m_patch_radius=3, m_search_area_radius=5, m_nms_radius=5, m_nms_scale_radius=0, m_th_saliency=250.0, m_kNN=4, m_scale_factor=1.25, m_n_scales=-1, m_compute_orientation=0)
tbmr  = cv2.xfeatures2d.TBMR.create(min_area=60, max_area_relative=0.01, scale_factor=1.25, n_scales=-1)
### descriptors 9
vgg   = cv2.xfeatures2d.VGG.create(isigma=1.4, img_normalize=True, use_scale_orientation=False, scale_factor=6.25, dsc_normalize=False)
daisy = cv2.xfeatures2d.DAISY.create(radius=15.0, q_radius=3, q_theta=8, q_hist=8, norm=cv2.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv2.xfeatures2d.FREAK.create(orientationNormalized=False,scaleNormalized=False,patternScale=22.0,nOctaves=4)
brief = cv2.xfeatures2d.BriefDescriptorExtractor.create(bytes=32, use_orientation=False)
lucid = cv2.xfeatures2d.LUCID.create(lucid_kernel=5,blur_kernel=0)
latch = cv2.xfeatures2d.LATCH.create(bytes=32,rotationInvariance=False,half_ssd_size=3,sigma=2.0)
beblid= cv2.xfeatures2d.BEBLID.create(scale_factor=6.25, n_bits=100)
teblid= cv2.xfeatures2d.TEBLID.create(scale_factor=6.25, n_bits=103)
boost = cv2.xfeatures2d.BoostDesc.create(use_scale_orientation=False, scale_factor=6.25)

# lists of the different detectors, descriptors and matching methods
# DetectDescript = list([sift, akaze, orb, brisk, kaze])
Detectors      = list([sift, akaze, orb, brisk, kaze, fast, mser, agast, gftt, star, hl, msd, tbmr]) # 13 detectors
Descriptors    = list([sift, akaze, orb, brisk, kaze, vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost]) # 14 descriptors
#matching       = list([cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_L2SQR, cv2.NORM_HAMMING]) # 4 matching methods
matching       = list([cv2.NORM_L2, cv2.NORM_HAMMING])

# ################ Scenario 1 (Intensity) ################
# print("Scenario 1 Intensity")
# val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
# val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
# nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images

# Rate_intensity = np.zeros((nbre_img, len(matching), len(Detectors), len(Descriptors)))
# img, List8Img = get_intensity_8Img(Image, val_b, val_c) # use the intensity change images (I+b and I*c)
# for k in range(nbre_img):
#     img2 = List8Img[k]
#     for c3 in range(len(matching)): # for bf.L2 mapping
#         for i in range(len(Detectors)):
#             method_dtect = Detectors[i]
#             keypoints1 = method_dtect.detect(img, None)
#             keypoints2 = method_dtect.detect(img2, None)
#             for j in range(len(Descriptors)):
#                 method_dscrpt = Descriptors[j]
#                 try:
#                     descriptors1 = method_dscrpt.compute(img, keypoints1)[1]
#                     descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
#                     print("Scenario 1 Intensity: image ", k, " Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
#                     Rate_intensity[k, c3, i, j] = evaluate_scenario_1(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
#                 except Exception as e:
#                     print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
#                     Rate_intensity[k, c3, i, j] = None
# # export numpy arrays
# np.save(basedir + 'arrays/Rate_intensity.npy', Rate_intensity)
# ##########################################################

# ################ Scenario 2: Scale ################
# print("Scenario 2 Scale")
# scale = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # s ∈]1.1 : 0.2 : 2.3]

# Rate_scale = np.zeros((len(scale), len(matching), len(Detectors), len(Descriptors)))
# for s in range(len(scale)): # for the 7 scale images
#     img = get_cam_scale(Image, scale[s])#[0] # image I
#     for c3 in range(len(matching)): 
#         for i in range(len(Detectors)):
#             method_dtect = Detectors[i]
#             keypoints1 = method_dtect.detect(img[0], None)
#             keypoints2 = method_dtect.detect(img[1], None)
#             for j in range(len(Descriptors)):
#                 method_dscrpt = Descriptors[j]
#                 try:
#                     descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
#                     descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1]
#                     print("Scenario 2 Scale: image ", k, " Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
#                     Rate_scale[s, c3, i, j] = evaluate_scenario_2(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3], scale[s])
#                 except Exception as e:
#                     print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
#                     Rate_scale[s, c3, i, j] = None
# # export numpy arrays
# np.save(basedir + 'arrays/Rate_scale.npy', Rate_scale)
# ##########################################################

# ################ Scenario 3: Rotation ################
# print("Scenario 3 Rotation")
# rot = [15, 30, 45, 60, 75, 90] # r ∈ [15 : 15 : 90

# Rate_rot = np.zeros((len(rot), len(matching), len(Detectors), len(Descriptors)))
# for r in range(len(rot)):
#     rot_matrix, img = get_cam_rot(Image, rot[r])
#     for c3 in range(len(matching)): # for bf.L1 and bf.L2 mapping
#         for i in range(len(Detectors)):
#             method_dtect = Detectors[i]
#             keypoints1 = method_dtect.detect(img[0], None)
#             keypoints2 = method_dtect.detect(img[1], None)
#             for j in range(len(Descriptors)):
#                 method_dscrpt = Descriptors[j]
#                 try:
#                     descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
#                     descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1]
#                     print("Scenario 3 Rotation: image ", k, " Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
#                     Rate_rot[r, c3, i, j] = evaluate_scenario_3(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3], rot[r], rot_matrix)
#                 except Exception as e:
#                     print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
#                     Rate_rot[r, c3, i, j] = None
# # export numpy arrays
# np.save(basedir + 'arrays/Rate_rot.npy', Rate_rot)
# ##########################################################

################ Scenario 4: graf ############################
print("Scenario 4 graf")
# Read the images
folder = '/graf'
img1 = cv2.imread(basedir + folder + '/img1.jpg')
img2 = cv2.imread(basedir + folder + '/img2.jpg')
img3 = cv2.imread(basedir + folder + '/img3.jpg')
img4 = cv2.imread(basedir + folder + '/img4.jpg')
img5 = cv2.imread(basedir + folder + '/img5.jpg')
img6 = cv2.imread(basedir + folder + '/img6.jpg')

# Detect the keypoints and compute the descriptors with the different detectors and descriptors
Rate_graf = np.zeros((6, len(matching), len(Detectors), len(Descriptors)))
for g in range(6): # for the 6 images (img1, img2, img3, img4, img5, img6
    for c3 in range(len(matching)): # for bf.L1 and bf.L2 mapping
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img1, None)
            keypoints2 = method_dtect.detect(img2, None)
            keypoints3 = method_dtect.detect(img3, None)
            keypoints4 = method_dtect.detect(img4, None)
            keypoints5 = method_dtect.detect(img5, None)
            keypoints6 = method_dtect.detect(img6, None)
            print("Detector ", i, " is calculated for all images")
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img1, keypoints1)[1]
                    descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
                    descriptors3 = method_dscrpt.compute(img3, keypoints3)[1]
                    descriptors4 = method_dscrpt.compute(img4, keypoints4)[1]
                    descriptors5 = method_dscrpt.compute(img5, keypoints5)[1]
                    descriptors6 = method_dscrpt.compute(img6, keypoints6)[1]
                    print("Descriptor ", j, " is calculated for all images")
                    Rate_graf[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                    print("Scenario 4 graf:", g, "Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                    Rate_graf[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints3, descriptors1, descriptors3, matching[c3])
                    print("Scenario 4 graf:", g, "Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                    Rate_graf[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints4, descriptors1, descriptors4, matching[c3])
                    print("Scenario 4 graf:", g, "Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                    Rate_graf[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints5, descriptors1, descriptors5, matching[c3])
                    print("Scenario 4 graf:", g, "Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                    Rate_graf[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints6, descriptors1, descriptors6, matching[c3])
                    print("Scenario 4 graf:", g, "Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                except Exception as e:
                    print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
                    Rate_graf[g, c3, i, j] = None
                    print("Rate_graf[g, c3, i, j] = None çalıştı")
# export numpy arrays
np.save(basedir + 'arrays/Rate_graf.npy', Rate_graf)
##########################################################

################ Scenario 5: wall ############################
print("Scenario 5 wall")
# Read the images
folder = '/wall'
img1 = cv2.imread(basedir + folder + '/img1.jpg')
img2 = cv2.imread(basedir + folder + '/img2.jpg')
img3 = cv2.imread(basedir + folder + '/img3.jpg')
img4 = cv2.imread(basedir + folder + '/img4.jpg')
img5 = cv2.imread(basedir + folder + '/img5.jpg')
img6 = cv2.imread(basedir + folder + '/img6.jpg')

# Detect the keypoints and compute the descriptors with the different detectors and descriptors
Rate_wall = np.zeros((6, len(matching), len(Detectors), len(Descriptors)))
for g in range(6): # for the 6 images (img1, img2, img3, img4, img5, img6
    for c3 in range(len(matching)): # for bf.L1 and bf.L2 mapping
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img1, None)
            keypoints2 = method_dtect.detect(img2, None)
            keypoints3 = method_dtect.detect(img3, None)
            keypoints4 = method_dtect.detect(img4, None)
            keypoints5 = method_dtect.detect(img5, None)
            keypoints6 = method_dtect.detect(img6, None)
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img1, keypoints1)[1]
                    descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
                    descriptors3 = method_dscrpt.compute(img3, keypoints3)[1]
                    descriptors4 = method_dscrpt.compute(img4, keypoints4)[1]
                    descriptors5 = method_dscrpt.compute(img5, keypoints5)[1]
                    descriptors6 = method_dscrpt.compute(img6, keypoints6)[1]
                    Rate_wall[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                    Rate_wall[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints3, descriptors1, descriptors3, matching[c3])
                    Rate_wall[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints4, descriptors1, descriptors4, matching[c3])
                    Rate_wall[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints5, descriptors1, descriptors5, matching[c3])
                    Rate_wall[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints6, descriptors1, descriptors6, matching[c3])
                    print("Scenario 5 wall: Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                except Exception as e:
                    print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
                    Rate_wall[g, c3, i, j] = None
# export numpy arrays
np.save(basedir + 'arrays/Rate_wall.npy', Rate_wall)
##########################################################

################ Scenario 6: trees ############################
print("Scenario 6 trees")
# Read the images
folder = '/trees'
img1 = cv2.imread(basedir + folder + '/img1.jpg')
img2 = cv2.imread(basedir + folder + '/img2.jpg')
img3 = cv2.imread(basedir + folder + '/img3.jpg')
img4 = cv2.imread(basedir + folder + '/img4.jpg')
img5 = cv2.imread(basedir + folder + '/img5.jpg')
img6 = cv2.imread(basedir + folder + '/img6.jpg')

# Detect the keypoints and compute the descriptors with the different detectors and descriptors
Rate_trees = np.zeros((6, len(matching), len(Detectors), len(Descriptors)))
for g in range(6): # for the 6 images (img1, img2, img3, img4, img5, img6
    for c3 in range(len(matching)): # for bf.L1 and bf.L2 mapping
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img1, None)
            keypoints2 = method_dtect.detect(img2, None)
            keypoints3 = method_dtect.detect(img3, None)
            keypoints4 = method_dtect.detect(img4, None)
            keypoints5 = method_dtect.detect(img5, None)
            keypoints6 = method_dtect.detect(img6, None)
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img1, keypoints1)[1]
                    descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
                    descriptors3 = method_dscrpt.compute(img3, keypoints3)[1]
                    descriptors4 = method_dscrpt.compute(img4, keypoints4)[1]
                    descriptors5 = method_dscrpt.compute(img5, keypoints5)[1]
                    descriptors6 = method_dscrpt.compute(img6, keypoints6)[1]
                    Rate_trees[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                    Rate_trees[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints3, descriptors1, descriptors3, matching[c3])
                    Rate_trees[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints4, descriptors1, descriptors4, matching[c3])
                    Rate_trees[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints5, descriptors1, descriptors5, matching[c3])
                    Rate_trees[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints6, descriptors1, descriptors6, matching[c3])
                    print("Scenario 6 trees: Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                except Exception as e:
                    print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
                    Rate_trees[g, c3, i, j] = None
# export numpy arrays
np.save(basedir + 'arrays/Rate_trees.npy', Rate_trees)
##########################################################

################ Scenario 7: bikes ############################
print("Scenario 7 bikes")
# Read the images
folder = '/bikes'
img1 = cv2.imread(basedir + folder + '/img1.jpg')
img2 = cv2.imread(basedir + folder + '/img2.jpg')
img3 = cv2.imread(basedir + folder + '/img3.jpg')
img4 = cv2.imread(basedir + folder + '/img4.jpg')
img5 = cv2.imread(basedir + folder + '/img5.jpg')
img6 = cv2.imread(basedir + folder + '/img6.jpg')

# Detect the keypoints and compute the descriptors with the different detectors and descriptors
Rate_bikes = np.zeros((6, len(matching), len(Detectors), len(Descriptors)))
for g in range(6): # for the 6 images (img1, img2, img3, img4, img5, img6
    for c3 in range(len(matching)): # for bf.L1 and bf.L2 mapping
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img1, None)
            keypoints2 = method_dtect.detect(img2, None)
            keypoints3 = method_dtect.detect(img3, None)
            keypoints4 = method_dtect.detect(img4, None)
            keypoints5 = method_dtect.detect(img5, None)
            keypoints6 = method_dtect.detect(img6, None)
            for j in range(len(Descriptors)):
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img1, keypoints1)[1]
                    descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
                    descriptors3 = method_dscrpt.compute(img3, keypoints3)[1]
                    descriptors4 = method_dscrpt.compute(img4, keypoints4)[1]
                    descriptors5 = method_dscrpt.compute(img5, keypoints5)[1]
                    descriptors6 = method_dscrpt.compute(img6, keypoints6)[1]
                    Rate_bikes[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                    Rate_bikes[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints3, descriptors1, descriptors3, matching[c3])
                    Rate_bikes[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints4, descriptors1, descriptors4, matching[c3])
                    Rate_bikes[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints5, descriptors1, descriptors5, matching[c3])
                    Rate_bikes[g, c3, i, j] = evaluate_scenario_4(keypoints1, keypoints6, descriptors1, descriptors6, matching[c3])
                    print("Scenario 7 bikes: Detector ", i, " Descriptor ", j, " Matching ", matching[c3], " is calculated")
                except Exception as e:
                    print("Combination of detector", Detectors[i], ", descriptor ", Descriptors[j], " and matching", matching[c3], "is not possible.")
                    Rate_bikes[g, c3, i, j] = None
# export numpy arrays
np.save(basedir + 'arrays/Rate_bikes.npy', Rate_bikes)
##########################################################