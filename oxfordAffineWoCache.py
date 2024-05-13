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
kaze   = cv2.KAZE_create(extended=False, upright=False, threshold=0.01, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)

### detectors 9
fast  = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8, threshold=20)
mser  = cv2.MSER_create(delta=5, min_area=60, max_area=14400, max_variation=0.25, min_diversity=0.90, max_evolution=20, area_threshold=1.01, min_margin=0.003, edge_blur_size=5)
agast = cv2.AgastFeatureDetector_create(threshold=5,nonmaxSuppression=True,type=cv2.AGAST_FEATURE_DETECTOR_AGAST_7_12D)
gftt  = cv2.GFTTDetector_create(qualityLevel=0.5, minDistance=20.0, blockSize=3, useHarrisDetector=False, k=0.04, maxCorners=2000)
gftt_harris = cv2.GFTTDetector_create(qualityLevel=0.5, minDistance=20.0, blockSize=3, useHarrisDetector=True, k=0.04, maxCorners=2000) 
star  = cv2.xfeatures2d.StarDetector_create(maxSize=20, responseThreshold=5, lineThresholdProjected=100, lineThresholdBinarized=30, suppressNonmaxSize=3)
hl    = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=4, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=2000, num_layers=4)
msd   = cv2.xfeatures2d.MSDDetector_create(m_patch_radius=3, m_search_area_radius=5, m_nms_radius=5, m_nms_scale_radius=0, m_th_saliency=250.0, m_kNN=4, m_scale_factor=1.25, m_n_scales=-1, m_compute_orientation=0)
tbmr  = cv2.xfeatures2d.TBMR_create(min_area=40, max_area_relative=0.01, scale_factor=1.25, n_scales=-1)

### descriptors 9
vgg675 = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=6.75, dsc_normalize=False) # for SIFT
vgg625 = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=6.25, dsc_normalize=False) # for KAZE, SURF
vgg500 = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=5.00, dsc_normalize=False) # for AKAZE, MSD, AGAST, FAST, BRISK
vgg075 = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=0.75, dsc_normalize=False) # for ORB
daisy = cv2.xfeatures2d.DAISY_create(radius=15, q_radius=3, q_theta=8, q_hist=8, norm=cv2.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv2.xfeatures2d.FREAK_create(orientationNormalized=True, scaleNormalized=False, patternScale=22.0, nOctaves=3)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=True)
lucid = cv2.xfeatures2d.LUCID_create(lucid_kernel=3, blur_kernel=6)
latch = cv2.xfeatures2d.LATCH_create(bytes=2, rotationInvariance=True, half_ssd_size=1, sigma=1.4)
beblid675 = cv2.xfeatures2d.BEBLID_create(scale_factor=6.75, n_bits=100) #for SIFT
beblid625 = cv2.xfeatures2d.BEBLID_create(scale_factor=6.25, n_bits=100) #for KAZE, SURF
beblid500 = cv2.xfeatures2d.BEBLID_create(scale_factor=5.00, n_bits=100) #for AKAZE, MSD, AGAST, FAST, BRISK
beblid100 = cv2.xfeatures2d.BEBLID_create(scale_factor=1.00, n_bits=100) # for ORB
teblid675 = cv2.xfeatures2d.TEBLID_create(scale_factor=6.75, n_bits=102) #for SIFT
teblid625 = cv2.xfeatures2d.TEBLID_create(scale_factor=6.25, n_bits=102) #for KAZE, SURF
teblid500 = cv2.xfeatures2d.TEBLID_create(scale_factor=5.00, n_bits=102) # for AKAZE, MSD, AGAST, FAST, BRISK
teblid100 = cv2.xfeatures2d.TEBLID_create(scale_factor=1.00, n_bits=102) # ORB
boost675 = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=6.75) #for SIFT
boost625 = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=6.25) #for KAZE, SURF 
boost500 = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=5.00) #for AKAZE, MSD, AGAST, FAST, BRISK
boost150 = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=1.50) #default in original implementation
boost075 = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=0.75) #for ORB

Detectors      = list([sift, akaze, orb, brisk, kaze,
                       fast, mser, agast, gftt, gftt_harris, star, hl, msd, tbmr])
Descriptors    = list([sift, akaze, orb, brisk, kaze,
                       vgg675, vgg625, vgg500, vgg075,
                       daisy, freak, brief, lucid, latch,
                       beblid675, beblid625, beblid500, beblid100,
                       teblid675, teblid625, teblid500, teblid100,
                       boost675, boost625, boost500, boost150, boost075]) 
matching       = list([cv2.NORM_L2, cv2.NORM_HAMMING])
matcher        = 0 # 0: Brute-force matcher, 1: Flann-based matcher
a = 100 #i
b = 100 #j
########################################################
def executeScenarios(folder):
    print(time.ctime())
    print(f"Folder: {folder}")
    if a == 100 and b == 100:
        Rate      = np.zeros((5, len(matching), len(Detectors), len(Descriptors)))
        Exec_time = np.zeros((6, len(matching), len(Detectors), len(Descriptors), 3))
    else:
        Rate      = np.load(f"{maindir}/arrays/Rate_{folder}.npy")
        Exec_time = np.load(f"{maindir}/arrays/Exec_time_{folder}.npy")
    
    img = [cv2.imread(f"{datasetdir}/{folder}/img{i}.jpg") for i in range(1, 7)]
    
    for k in range(1, len(img)):
        for i in range(len(Detectors)):
            if i == a or a == 100:
                method_dtect = Detectors[i]
                keypoints1 = method_dtect.detect(img[0], None)
                keypoints2 = method_dtect.detect(img[k], None)
                for j in range(len(Descriptors)):
                    if j == b or b == 100:
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