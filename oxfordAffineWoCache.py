import cv2
import numpy as np
import time, os

maindir = os.path.abspath(os.path.dirname(__file__))
datasetdir = "./oxfordAffine"

def match_with_bf_ratio_test(matcher, Dspt1, Dspt2, norm_type, threshold_ratio=0.8):
    if matcher == 0: # Brute-force matcher
        bf = cv2.BFMatcher(norm_type, crossCheck=True) 
        matches = bf.match(Dspt1, Dspt2)
    else: # Flann-based matcher
        if norm_type == cv2.NORM_L2:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        elif norm_type == cv2.NORM_HAMMING:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(Dspt1, Dspt2)
        
    good_matches = []
    for m,n in matches:
        if m.distance < threshold_ratio * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key = lambda x:x.distance)
    match_rate = len(good_matches) / len(matches) * 100
    return match_rate, good_matches
# ................................................................................

def evaluate_with_fundamentalMat_and_XSAC(matcher, KP1, KP2, Dspt1, Dspt2, norm_type):
    if matcher == 0: # Brute-force matcher
        bf = cv2.BFMatcher(norm_type, crossCheck=True) 
        matches = bf.match(Dspt1, Dspt2)
    else: # Flann-based matcher
        if norm_type == cv2.NORM_L2:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
        elif norm_type == cv2.NORM_HAMMING:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(Dspt1, Dspt2)
                
    points1 = np.array([KP1[match.queryIdx].pt for match in matches], dtype=np.float32)
    points2 = np.array([KP2[match.trainIdx].pt for match in matches], dtype=np.float32)
    
    h, mask = cv2.findFundamentalMat(points1, points2, cv2.USAC_MAGSAC)
    inliers = [matches[i] for i in range(len(matches)) if mask[i] == 1]

    inliers_percentage = (len(inliers) / len(matches)) * 100
    return inliers_percentage, inliers
# ................................................................................

### detectors/descriptors 5
sift   = cv2.SIFT_create(nfeatures=2000, nOctaveLayers=3, contrastThreshold=0.1, edgeThreshold=10.0, sigma=1.6) #best with layer=3 contrastThreshold=0.1 
akaze  = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.01, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
orb    = cv2.ORB_create(nfeatures=2000, scaleFactor=1.1, nlevels=6, edgeThreshold=60, firstLevel=1, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=60, fastThreshold=60)
brisk  = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
kaze   = cv2.KAZE_create(extended=False, upright=False, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2) #, threshold=0.01)

### detectors 9
fast  = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
mser  = cv2.MSER_create(delta=5, min_area=60, max_area=14400, max_variation=0.25, area_threshold=1.01, min_margin=0.003, edge_blur_size=5) #, min_diversity=0.95, max_evolution=10)
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
#                       0       1    2     3      4     5    6      7      8      9          10   11  12    13
Descriptors    = list([sift, akaze, orb, brisk, kaze, vgg, daisy, freak, brief, lucid, latch, beblid, teblid, boost]) # 14 descriptors
#                       0       1    2     3      4    5     6      7      8      9      10     11      12     13
matching       = list([cv2.NORM_L2, cv2.NORM_HAMMING])
matcher        = 0 # 0: Brute-force matcher, 1: Flann-based matcher
a = 20 #i
b = 20 #j
########################################################
def executeScenarios(folder):
    print(time.ctime())
    print(f"Folder: {folder}")
    if a == 20 and b == 20:
        Rate      = np.zeros((5, len(matching), len(Detectors), len(Descriptors)))
        Exec_time = np.zeros((6, len(matching), len(Detectors), len(Descriptors), 3))
    else:
        Rate      = np.load(f"{maindir}/arrays/Rate_{folder}.npy")
        Exec_time = np.load(f"{maindir}/arrays/Exec_time_{folder}.npy")
    
    img = [cv2.imread(f"{datasetdir}/{folder}/img{i}.jpg") for i in range(1, 7)]
    
    for k in range(1, len(img)):
        for i in range(len(Detectors)):
            if i == a or a == 20:
                method_dtect = Detectors[i]
                keypoints1 = method_dtect.detect(img[0], None)
                keypoints2 = method_dtect.detect(img[k], None)
                for j in range(len(Descriptors)):
                    if j == b or b == 20:
                        for c3 in range(len(matching)):
                            method_dscrpt = Descriptors[j]
                            try:
                                descriptors1 = method_dscrpt.compute(img[0], keypoints1)[1]
                                descriptors2 = method_dscrpt.compute(img[k], keypoints2)[1]
                            except:
                                continue
                            try:
                                Rate[k-1, c3, i, j], _ = evaluate_with_fundamentalMat_and_XSAC(matcher, keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                            except:
                                Rate[k-1, c3, i, j] = None
                                continue
                    else:
                        continue
            else:
                continue
    np.save(f"{maindir}/arrays/Rate_{folder}.npy",      Rate)
    np.save(f"{maindir}/arrays/Exec_time_{folder}.npy", Exec_time)
########################################################
executeScenarios("graf")
executeScenarios("wall")
executeScenarios("trees")
executeScenarios("bikes")
executeScenarios("bark")
executeScenarios("boat")
executeScenarios("leuven")
executeScenarios("ubc")
########################################################
print(time.ctime())