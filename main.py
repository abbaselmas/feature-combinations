import cv2 # opencv
import numpy as np # For numerical calculations
import logging # For logging
import time
import os

# Creating logger
mylogs = logging.getLogger(__name__)
mylogs.setLevel(logging.INFO)
# Handler - 1
file = logging.FileHandler("log.txt")
fileformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
file.setLevel(logging.INFO)
file.setFormatter(fileformat)
# Handler - 2
stream = logging.StreamHandler()
streamformat = logging.Formatter("%(levelname)s:%(module)s:%(message)s")
stream.setLevel(logging.INFO)
stream.setFormatter(streamformat)
# Adding all handlers to the logs
mylogs.addHandler(file)
mylogs.addHandler(stream)

maindir = os.path.abspath(os.path.dirname(__file__))
datasetdir = "./oxfordAffine"
folder = "/graf"
picture = "/img1.jpg"
data = datasetdir + folder + picture

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
        filename = f"{maindir}/intensity/image_{i}.png"  # You can change the format and naming convention as needed
        cv2.imwrite(filename, img)
    return Img, List8Img
# ................................................................................

## Scenario 2 (Scale): Function that takes as input the index of the camera, the index of the image n, and a scale, it returns
#                      a couple (I, Iscale). In the following, we will work with 7 images with a scale change Is : s ∈]1.1 : 0.2 : 2.3].
def get_cam_scale(Img, s):
    ImgScale = cv2.resize(Img, (0, 0), fx=s, fy=s, interpolation = cv2.INTER_NEAREST) # opencv resize function with INTER_NEAREST interpolation
    I_Is = list([Img, ImgScale]) # list of 2 images (original image and scaled image)
    # Save the images to disk
    filename = f"{maindir}/scale/image_{s}.png"
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
    filename = f"{maindir}/rotation/image_{r}.png"  # You can change the format and naming convention as needed
    cv2.imwrite(filename, rotated_image)

    return rotation_mat, couple_I_Ir  # it also returns the rotation matrix for further use in the rotation evaluation function
# ................................................................................

import cv2
import numpy as np

def evaluate_with_geometric_verification(KP1, KP2, Dspt1, Dspt2, norm_type, threshold=10):
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.match(Dspt1, Dspt2)
    pts1 = np.float32([KP1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([KP2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    pts1_transformed = cv2.perspectiveTransform(pts1, H)
    distances = np.linalg.norm(pts1_transformed - pts2, axis=2)
    inliers = np.sum(distances < threshold)
    match_rate = (inliers / len(matches)) * 100
    return match_rate

def evaluate_with_descriptor_distance(KP1, KP2, Dspt1, Dspt2, norm_type, max_distance_factor=1.0): #TODO: 100 constant to be parameterized or adjusted
    valid_match_count = 0
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.match(Dspt1, Dspt2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Determine the appropriate maximum distance threshold based on the norm type
    if norm_type == cv2.NORM_HAMMING:
        max_distance = max_distance_factor * len(Dspt1[0])  # Number of bits in the descriptor
    else:  # Assuming NORM_L2
        max_distance = max_distance_factor * 100  # Adjust as needed based on descriptor space
    # Iterate over matches until the distance exceeds the maximum allowed distance
    for match in matches:
        if match.distance <= max_distance:
            valid_match_count += 1
        else:
            break  # No need to continue evaluating matches if distance exceeds the threshold
    match_rate = valid_match_count / max(len(KP1), len(KP2)) * 100
    return match_rate

## Evaluation of scenario 4: graf: Function that takes as input the keypoints, the descriptors (of 2 images),
#                            the type of matching, it returns the percentage of correct matched points
def evaluate_ratio(Dspt1, Dspt2, match_method):
    bf = cv2.BFMatcher(normType=match_method, crossCheck=False)
    matches = bf.knnMatch(Dspt1,Dspt2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
    return len(good) / len(matches) * 100

### detectors/descriptors 5
sift  = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=100.0, sigma=1.6)
akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_KAZE, descriptor_size=0, descriptor_channels=3, threshold=0.00005, nOctaves=4, nOctaveLayers=3, diffusivity=cv2.KAZE_DIFF_PM_G1)
orb   = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=4, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=12)
brisk = cv2.BRISK_create(thresh=50, octaves=1, patternScale=1.2)
kaze  = cv2.KAZE_create(extended=False, upright=False, threshold=0.00005,  nOctaves=4, nOctaveLayers=3, diffusivity=cv2.KAZE_DIFF_PM_G2)
### detectors 8
fast  = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True, type=cv2.FastFeatureDetector_TYPE_9_16)
mser  = cv2.MSER_create(delta=1, min_area=30, max_area=1440, max_variation=0.025, min_diversity=0.8, max_evolution=200, area_threshold=1.01, min_margin=0.003, edge_blur_size=3)
agast = cv2.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv2.AgastFeatureDetector_OAST_9_16)
gftt  = cv2.GFTTDetector_create(maxCorners=20000, qualityLevel=0.002, minDistance=1.0, blockSize=3, useHarrisDetector=False, k=0.04)
star  = cv2.xfeatures2d.StarDetector_create(maxSize=15, responseThreshold=1, lineThresholdProjected=10, lineThresholdBinarized=8, suppressNonmaxSize=3)
hl    = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=6, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=20000, num_layers=4)
msd   = cv2.xfeatures2d.MSDDetector_create(m_patch_radius=3, m_search_area_radius=5, m_nms_radius=5, m_nms_scale_radius=0, m_th_saliency=250.0, m_kNN=4, m_scale_factor=1.25, m_n_scales=-1, m_compute_orientation=0)
tbmr  = cv2.xfeatures2d.TBMR_create(min_area=60, max_area_relative=0.01, scale_factor=1.25, n_scales=-1)
### descriptors 9
vgg   = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=5.00, dsc_normalize=False)
daisy = cv2.xfeatures2d.DAISY_create(radius=15.0, q_radius=3, q_theta=8, q_hist=8, norm=cv2.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv2.xfeatures2d.FREAK_create(orientationNormalized=False,scaleNormalized=False,patternScale=22.0,nOctaves=4)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)
lucid = cv2.xfeatures2d.LUCID_create(lucid_kernel=1,blur_kernel=2)
latch = cv2.xfeatures2d.LATCH_create(bytes=32,rotationInvariance=False,half_ssd_size=3,sigma=2.0)
beblid= cv2.xfeatures2d.BEBLID_create(scale_factor=5.00, n_bits=101)
teblid= cv2.xfeatures2d.TEBLID_create(scale_factor=5.00, n_bits=102)
boost = cv2.xfeatures2d.BoostDesc_create(desc=300, use_scale_orientation=False, scale_factor=5.00)

# DetectDescript = list([sift, akaze, orb, brisk, kaze])
# binary_detectors = [orb, brisk]
# non_binary_detectors = [sift, akaze, kaze, fast, mser, agast, gftt, star, hl, msd, tbmr]
# binary_descriptors = [orb, brisk, freak, brief, latch, beblid, teblid, boost]
# non_binary_descriptors = [sift, akaze, kaze, vgg, daisy, lucid]

Detectors      = list([sift, akaze, orb, brisk, kaze, fast, mser, agast, gftt, star, hl, msd, tbmr]) # 13 detectors
Descriptors    = list([sift, akaze, orb, brisk, kaze, vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost]) # 14 descriptors
matching       = list([cv2.NORM_L2, cv2.NORM_HAMMING])

################ Scenario 1 (Intensity) ################
print("Scenario 1 Intensity")
val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images

Rate_intensity      = np.zeros((nbre_img, len(matching), len(Detectors), len(Descriptors)))
Exec_time_intensity = np.zeros((nbre_img, len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)
img, List8Img = get_intensity_8Img(Image, val_b, val_c) # use the intensity change images (I+b and I*c)
for k in range(nbre_img):
    img2 = List8Img[k]
    for c3 in range(len(matching)): # for bf.L2 mapping
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img, None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img2, None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_intensity[k, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_intensity[k, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img, keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
                    end_time = time.time()
                    Exec_time_intensity[k, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_intensity[k, c3, i, j, 1])
                    start_time = time.time()
                    Rate_intensity[k, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_intensity[k, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 1 Intensity %s | Detector %s Descriptor %s Matching %s is calculated within %f", k, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_intensity[k, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_intensity[k, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_intensity.npy", Rate_intensity)
np.save(maindir + "/arrays/Exec_time_intensity.npy", Exec_time_intensity)
##########################################################

################ Scenario 2: Scale ################
print("Scenario 2 Scale")
scale = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # s ∈]1.1 : 0.2 : 2.3]

Rate_scale      = np.zeros((len(scale), len(matching), len(Detectors), len(Descriptors)))
Exec_time_scale = np.zeros((len(scale), len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)

for s in range(len(scale)): # for the 7 scale images
    img = get_cam_scale(Image, scale[s])#[0] # image I
    for c3 in range(len(matching)): 
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[1], None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_scale[s, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_scale[s, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1]
                    end_time = time.time()
                    Exec_time_scale[s, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_scale[s, c3, i, j, 1])
                    start_time = time.time()
                    Rate_scale[s, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_scale[s, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 2 Scale %s | Detector %s Descriptor %s Matching %s is calculated within %f", s, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_scale[s, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_scale[s, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_scale.npy", Rate_scale)
np.save(maindir + "/arrays/Exec_time_scale.npy", Exec_time_scale)
##########################################################

################ Scenario 3: Rotation ################
print("Scenario 3 Rotation")
rot = [15, 30, 45, 60, 75, 90] # r ∈ [15 : 15 : 90

Rate_rot       = np.zeros((len(rot), len(matching), len(Detectors), len(Descriptors)))
Exec_time_rot  = np.zeros((len(rot), len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)

for r in range(len(rot)):
    rot_matrix, img = get_cam_rot(Image, rot[r])
    for c3 in range(len(matching)): # for bf.L1 and bf.L2 mapping
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[1], None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_rot[r, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_rot[r, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1]
                    end_time = time.time()
                    Exec_time_rot[r, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_rot[r, c3, i, j, 1])
                    start_time = time.time()
                    Rate_rot[r, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_rot[r, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 3 Rotation %s | Detector %s Descriptor %s Matching %s is calculated within %f", r, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_rot[r, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_rot[r, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_rot.npy", Rate_rot)
np.save(maindir + "/arrays/Exec_time_rot.npy", Exec_time_rot)
##########################################################

################ Scenario 4: graf ############################
print("Scenario 4 graf")
folder = "/graf"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_graf       = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors)))
Exec_time_graf  = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)

for g in range(len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[g], None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_graf[g, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_graf[g, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[g], keypoints2)[1]
                    end_time = time.time()
                    Exec_time_graf[g, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_graf[g, c3, i, j, 1])
                    start_time = time.time()
                    Rate_graf[g, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_graf[g, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 4 graf %s | Detector %s Descriptor %s Matching %s is calculated within %f", g, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_graf[g, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_graf[g, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_graf.npy", Rate_graf)
np.save(maindir + "/arrays/Exec_time_graf.npy", Exec_time_graf)
##########################################################

################ Scenario 5: wall ############################
print("Scenario 5 wall")
# Read the images
folder = "/wall"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_wall       = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors)))
Exec_time_wall  = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)
for g in range(len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[g], None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_wall[g, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_graf[g, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[g], keypoints2)[1]
                    end_time = time.time()
                    Exec_time_wall[g, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_graf[g, c3, i, j, 1])
                    start_time = time.time()
                    Rate_wall[g, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_wall[g, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 5 wall %s | Detector %s Descriptor %s Matching %s is calculated within %f", g, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_wall[g, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_wall[g, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_wall.npy", Rate_wall)
np.save(maindir + "/arrays/Exec_time_wall.npy", Exec_time_wall)
##########################################################

################ Scenario 6: trees ############################
print("Scenario 6 trees")
# Read the images
folder = "/trees"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_trees       = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors)))
Exec_time_trees  = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)
for g in range(len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[g], None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_trees[g, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_graf[g, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[g], keypoints2)[1]
                    end_time = time.time()
                    Exec_time_trees[g, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_graf[g, c3, i, j, 1])
                    start_time = time.time()
                    Rate_trees[g, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_trees[g, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 6 trees %s | Detector %s Descriptor %s Matching %s is calculated within %f", g, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_trees[g, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_trees[g, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_trees.npy", Rate_trees)
np.save(maindir + "/arrays/Exec_time_trees.npy", Exec_time_trees)
##########################################################

################ Scenario 7: bikes ############################
print("Scenario 7 bikes")
# Read the images
folder = "/bikes"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_bikes       = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors)))
Exec_time_bikes  = np.zeros((len(img), len(matching), len(Detectors), len(Descriptors), 3))  # 3 for detect, compute, and evaluate_scenario (match)
for g in range(len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[g], None)
            end_time = time.time()
            for j in range(len(Descriptors)):
                Exec_time_bikes[g, c3, i, j, 0] = end_time - start_time
                mylogs.info("Detector %s is calculated for all images within %f", method_dtect.getDefaultName(), Exec_time_graf[g, c3, i, j, 0])
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[g], keypoints2)[1]
                    end_time = time.time()
                    Exec_time_bikes[g, c3, i, j, 1] = end_time - start_time
                    mylogs.info("Descriptor %s is calculated for all images within %f", method_dscrpt.getDefaultName(), Exec_time_graf[g, c3, i, j, 1])
                    start_time = time.time()
                    Rate_bikes[g, c3, i, j] = evaluate_ratio(descriptors1, descriptors2, matching[c3])
                    end_time = time.time()
                    Exec_time_bikes[g, c3, i, j, 2] = end_time - start_time
                    mylogs.info("Scenario 7 bikes %s | Detector %s Descriptor %s Matching %s is calculated within %f", g, method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3], Exec_time_bikes[g, c3, i, j, 2])
                except Exception as e:
                    mylogs.info("Combination of detector %s, descriptor %s and matching %s is not possible.", method_dtect.getDefaultName(), method_dscrpt.getDefaultName(), matching[c3])
                    Rate_bikes[g, c3, i, j] = None
# export numpy arrays
np.save(maindir + "/arrays/Rate_bikes.npy", Rate_bikes)
np.save(maindir + "/arrays/Exec_time_bikes.npy", Exec_time_bikes)
##########################################################