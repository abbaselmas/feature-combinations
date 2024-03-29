import cv2
import numpy as np
import time, os

maindir = os.path.abspath(os.path.dirname(__file__))
datasetdir = "./oxfordAffine"
folder = "/graf"
picture = "/img1.jpg"
data = datasetdir + folder + picture

Image = cv2.imread(data)
Image = np.array(Image)

val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images
scale = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # s ∈]1.1 : 0.2 : 2.3]
rot = [15, 30, 45, 60, 75, 90] # r ∈ [15 : 15 : 90

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
        filename = f"{maindir}/intensity/image_I+{val_b[i]}.png"
        cv2.imwrite(filename, List8Img[i])
    for j in range(len(val_c)): # for I ∗ c, with: c ∈ [0.7 : 0.2 : 1.3].
        I =  image * val_c[j]
        List8Img[j+4] = I.astype(int)
        List8Img[j+4][List8Img[j+4] > 255] = 255 # set pixels with intensity > 255 to 255
        List8Img[j+4][List8Img[j+4] < 0] = 0 # set the pixels with intensity < 0 to the value of 0
        List8Img[j+4] = np.array(List8Img[j+4], dtype=np.uint8) # transform image to uint8 (min value = 0, max value = 255)
        filename = f"{maindir}/intensity/image_Ix{val_c[j]}.png"
        cv2.imwrite(filename, List8Img[j+4])
    return Img, List8Img
## Scenario 2 (Scale): Function that takes as input the index of the camera, the index of the image n, and a scale, it returns a couple (I, Iscale). In the following, we will work with 7 images with a scale change Is : s ∈]1.1 : 0.2 : 2.3].
def get_cam_scale(Img, s):
    ImgScale = cv2.resize(Img, (0, 0), fx=s, fy=s, interpolation = cv2.INTER_NEAREST) # opencv resize function with INTER_NEAREST interpolation
    I_Is = list([Img, ImgScale]) # list of 2 images (original image and scaled image)
    # Save the images to disk
    filename = f"{maindir}/scale/image_{s}.png"
    cv2.imwrite(filename, ImgScale)
    return I_Is
## Scenario 3 (Rotation): Function that takes as input the index of the camera, the index of the image n, and a rotation angle, it returns a couple (I, Irot), and the rotation matrix. In the following, we will work with 9 images with a change of scale For an image I, we will create 9 images (I10, I20...I90) with change of rotation from 10 to 90 with a step of 10.
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

    return couple_I_Ir

def match_with_bf_ratio_test(Dspt1, Dspt2, norm_type, threshold_ratio=0.8):
    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    matches = bf.knnMatch(Dspt1,Dspt2,k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < threshold_ratio*n.distance:
            good_matches.append([m])
    good_matches = sorted(good_matches, key = lambda x:x[0].distance)
    match_rate = len(good_matches) / len(matches) * 100
    return match_rate, good_matches

def match_with_flannbased_NNDR(Dspt1, Dspt2, norm_type, threshold_ratio=0.8):
    if norm_type == cv2.NORM_L2:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
    elif norm_type == cv2.NORM_HAMMING:
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(Dspt1, Dspt2, 2)
    good_matches = []
    for m, n in matches:
        if m.distance < threshold_ratio * n.distance:
            good_matches.append([m])
    good_matches = sorted(good_matches, key=lambda x: x[0].distance)
    match_rate = len(good_matches) / len(matches) * 100
    return match_rate, good_matches

### detectors/descriptors 5
sift   = cv2.SIFT_create(nfeatures=2000, nOctaveLayers=3, contrastThreshold=0.1, edgeThreshold=10.0, sigma=1.6) #best with layer=3 contrastThreshold=0.1 
akaze  = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.01, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
orb    = cv2.ORB_create(nfeatures=2000, scaleFactor=1.1, nlevels=6, edgeThreshold=60, firstLevel=1, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=60, fastThreshold=60)
brisk  = cv2.BRISK_create(thresh=130, octaves=1, patternScale=1.1)
kaze   = cv2.KAZE_create(extended=False, upright=False, threshold=0.01,  nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)

### detectors 9
fast  = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
mser  = cv2.MSER_create(delta=5, min_area=60, max_area=14400, max_variation=0.25, min_diversity=0.95, max_evolution=10, area_threshold=1.01, min_margin=0.003, edge_blur_size=5)
agast = cv2.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv2.AGAST_FEATURE_DETECTOR_AGAST_7_12D)
gftt  = cv2.GFTTDetector_create(maxCorners=2000, qualityLevel=0.5, minDistance=20.0, blockSize=3, useHarrisDetector=False, k=0.04)
gftt_harris = cv2.GFTTDetector_create(maxCorners=2000, qualityLevel=0.5, minDistance=20.0, blockSize=3, useHarrisDetector=True, k=0.04)
star  = cv2.xfeatures2d.StarDetector_create(maxSize=20, responseThreshold=5, lineThresholdProjected=100, lineThresholdBinarized=30, suppressNonmaxSize=3)
hl    = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=4, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=2000, num_layers=4)
msd   = cv2.xfeatures2d.MSDDetector_create(m_patch_radius=3, m_search_area_radius=5, m_nms_radius=5, m_nms_scale_radius=0, m_th_saliency=250.0, m_kNN=4, m_scale_factor=1.25, m_n_scales=-1, m_compute_orientation=0)
tbmr  = cv2.xfeatures2d.TBMR_create(min_area=40, max_area_relative=0.01, scale_factor=1.25, n_scales=-1)

### descriptors 9
vgg   = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=6.75, dsc_normalize=False)
daisy = cv2.xfeatures2d.DAISY_create(radius=15, q_radius=3, q_theta=8, q_hist=8, norm=cv2.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv2.xfeatures2d.FREAK_create(orientationNormalized=True,scaleNormalized=False,patternScale=22.0,nOctaves=3)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=True)
lucid = cv2.xfeatures2d.LUCID_create(lucid_kernel=3,blur_kernel=6)
latch = cv2.xfeatures2d.LATCH_create(bytes=2,rotationInvariance=True,half_ssd_size=1,sigma=1.4)
beblid= cv2.xfeatures2d.BEBLID_create(scale_factor=6.75, n_bits=100)
teblid= cv2.xfeatures2d.TEBLID_create(scale_factor=6.75, n_bits=102)
boost = cv2.xfeatures2d.BoostDesc_create(desc=300, use_scale_orientation=True, scale_factor=0.75)

Detectors      = list([sift, akaze, orb, brisk, kaze, fast, mser, agast, gftt, gftt_harris, star, hl, msd, tbmr]) # 14 detectors
Descriptors    = list([sift, akaze, orb, brisk, kaze, vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost]) # 14 descriptors
matching       = list([cv2.NORM_L2, cv2.NORM_HAMMING])

################ Scenario 1 (Intensity) ################
print("Scenario 1 Intensity")
Rate_intensity      = np.zeros((nbre_img, len(matching), len(Detectors), len(Descriptors)))
Exec_time_intensity = np.zeros((nbre_img, len(matching), len(Detectors), len(Descriptors), 3))
img, List8Img = get_intensity_8Img(Image, val_b, val_c)
for k in range(nbre_img):
    img2 = List8Img[k]
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img, None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img2, None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_intensity[k, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img, keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img2, keypoints2)[1]
                    Exec_time_intensity[k, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_intensity[k, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_intensity[k, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_intensity[k, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_intensity[k, c3, i, j] = None
                    Exec_time_intensity[k, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img, keypoints1, img2, keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws/intensity/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_intensity[k, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_intensity.npy", Rate_intensity)
np.save(maindir + "/arrays/Exec_time_intensity.npy", Exec_time_intensity)
##########################################################

################ Scenario 2: Scale ################
print("Scenario 2 Scale")
Rate_scale      = np.zeros((len(scale), len(matching), len(Detectors), len(Descriptors)))
Exec_time_scale = np.zeros((len(scale), len(matching), len(Detectors), len(Descriptors), 3))
for k in range(len(scale)):
    img = get_cam_scale(Image, scale[k])
    for c3 in range(len(matching)): 
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[1], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_scale[k, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1]
                    Exec_time_scale[k, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_scale[k, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_scale[k, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_scale[k, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_scale[k, c3, i, j] = None
                    Exec_time_scale[k, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[1], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws/scale/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_scale[k, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_scale.npy", Rate_scale)
np.save(maindir + "/arrays/Exec_time_scale.npy", Exec_time_scale)
##########################################################

################ Scenario 3: Rotation ################
print("Scenario 3 Rotation")
Rate_rot       = np.zeros((len(rot), len(matching), len(Detectors), len(Descriptors)))
Exec_time_rot  = np.zeros((len(rot), len(matching), len(Detectors), len(Descriptors), 3))
for k in range(len(rot)):
    img = get_cam_rot(Image, rot[k])
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[1], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_rot[k, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[1], keypoints2)[1]
                    Exec_time_rot[k, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_rot[k, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_rot[k, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_rot[k, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_rot[k, c3, i, j] = None
                    Exec_time_rot[k, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[1], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws/rot/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_rot[k, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_rot.npy", Rate_rot)
np.save(maindir + "/arrays/Exec_time_rot.npy", Exec_time_rot)
##########################################################
"""
..#######..##.....##.########..#######..########..########.
.##.....##..##...##..##.......##.....##.##.....##.##.....##
.##.....##...##.##...##.......##.....##.##.....##.##.....##
.##.....##....###....######...##.....##.########..##.....##
.##.....##...##.##...##.......##.....##.##...##...##.....##
.##.....##..##...##..##.......##.....##.##....##..##.....##
..#######..##.....##.##........#######..##.....##.########.
"""
################ Scenario 4: graf ############################
print("Scenario 4 graf")
folder = "/graf"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_graf       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_graf  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_graf[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_graf[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_graf[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_graf[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_graf[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_graf[k-1, c3, i, j] = None
                    Exec_time_graf[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_graf[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_graf.npy", Rate_graf)
np.save(maindir + "/arrays/Exec_time_graf.npy", Exec_time_graf)
##########################################################

################ Scenario 5: wall ############################
print("Scenario 5 wall")
folder = "/wall"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_wall       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_wall  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_wall[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_wall[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_wall[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_wall[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_wall[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_wall[k-1, c3, i, j] = None
                    Exec_time_wall[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_wall[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_wall.npy", Rate_wall)
np.save(maindir + "/arrays/Exec_time_wall.npy", Exec_time_wall)
##########################################################

################ Scenario 6: trees ############################
print("Scenario 6 trees")
folder = "/trees"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_trees       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_trees  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_trees[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_trees[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_trees[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_trees[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_trees[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_trees[k-1, c3, i, j] = None
                    Exec_time_trees[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_trees[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_trees.npy", Rate_trees)
np.save(maindir + "/arrays/Exec_time_trees.npy", Exec_time_trees)
##########################################################

################ Scenario 7: bikes ############################
print("Scenario 7 bikes")
folder = "/bikes"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_bikes       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_bikes  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_bikes[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_bikes[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_bikes[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_bikes[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_bikes[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_bikes[k-1, c3, i, j] = None
                    Exec_time_bikes[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_bikes[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_bikes.npy", Rate_bikes)
np.save(maindir + "/arrays/Exec_time_bikes.npy", Exec_time_bikes)
##########################################################
"""
..#######..##.....##.########..#######..########..########.....########..#######..########..#######.
.##.....##..##...##..##.......##.....##.##.....##.##.....##....##.......##.....##.##....##.##.....##
.##.....##...##.##...##.......##.....##.##.....##.##.....##....##.......##............##...##.....##
.##.....##....###....######...##.....##.########..##.....##....#######..########.....##.....#######.
.##.....##...##.##...##.......##.....##.##...##...##.....##..........##.##.....##...##.....##.....##
.##.....##..##...##..##.......##.....##.##....##..##.....##....##....##.##.....##...##.....##.....##
..#######..##.....##.##........#######..##.....##.########......######...#######....##......#######.
"""
################ Scenario 8: bark ############################
print("Scenario 8 bark")
folder = "/bark"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_bark       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_bark  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_bark[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_bark[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_bark[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_bark[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_bark[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_bark[k-1, c3, i, j] = None
                    Exec_time_bark[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_bark[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_bark.npy", Rate_bark)
np.save(maindir + "/arrays/Exec_time_bark.npy", Exec_time_bark)
##########################################################

################ Scenario 9: boat ############################
print("Scenario 9 boat")
folder = "/boat"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_boat       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_boat  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_boat[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_boat[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_boat[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_boat[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_boat[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_boat[k-1, c3, i, j] = None
                    Exec_time_boat[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_boat[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_boat.npy", Rate_boat)
np.save(maindir + "/arrays/Exec_time_boat.npy", Exec_time_boat)
##########################################################

################ Scenario 10: leuven ############################
print("Scenario 10 leuven")
folder = "/leuven"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_leuven       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_leuven  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_leuven[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_leuven[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_leuven[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_leuven[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_leuven[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_leuven[k-1, c3, i, j] = None
                    Exec_time_leuven[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_leuven[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_leuven.npy", Rate_leuven)
np.save(maindir + "/arrays/Exec_time_leuven.npy", Exec_time_leuven)
##########################################################

################ Scenario 11: ubc ############################
print("Scenario 11 ubc")
folder = "/ubc"
img = [cv2.imread(datasetdir + folder + f"/img{i}.jpg") for i in range(1, 7)]

Rate_ubc       = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors)))
Exec_time_ubc  = np.zeros((len(img)-1, len(matching), len(Detectors), len(Descriptors), 3))
for k in range(1, len(img)):
    for c3 in range(len(matching)):
        for i in range(len(Detectors)):
            method_dtect = Detectors[i]
            keypoints1 = method_dtect.detect(img[0], None)
            start_time = time.time()
            keypoints2 = method_dtect.detect(img[k], None)
            detector_time = time.time() - start_time
            for j in range(len(Descriptors)):
                Exec_time_ubc[k-1, c3, i, j, 0] = detector_time
                method_dscrpt = Descriptors[j]
                try:
                    descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                    start_time = time.time()
                    descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                    Exec_time_ubc[k-1, c3, i, j, 1] = time.time() - start_time
                except:
                    Exec_time_ubc[k-1, c3, i, j, 1] = None
                    continue
                try:
                    start_time = time.time()
                    Rate_ubc[k-1, c3, i, j], good_matches = match_with_flannbased_NNDR(descriptors1, descriptors2, matching[c3])
                    Exec_time_ubc[k-1, c3, i, j, 2] = time.time() - start_time
                except:
                    Rate_ubc[k-1, c3, i, j] = None
                    Exec_time_ubc[k-1, c3, i, j, 2] = None
                    continue
                # # draw matches
                # img_matches = cv2.drawMatchesKnn(img[0], keypoints1, img[k], keypoints2, good_matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                # filename = f"{maindir}/draws{folder}/{k}_{i}_{j}_{matching[c3]}_R_{int(Rate_ubc[k-1, c3, i, j])}.png"
                # cv2.imwrite(filename, img_matches)
np.save(maindir + "/arrays/Rate_ubc.npy", Rate_ubc)
np.save(maindir + "/arrays/Exec_time_ubc.npy", Exec_time_ubc)
##########################################################
