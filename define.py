import cv2
from plotly.colors import sample_colorscale
import numpy as np

val_b    = np.array([-30, -10, 10, 30])     # b ∈ [−30 : 20 : +30]
val_c    = np.array([0.7, 0.9, 1.1, 1.3])   # c ∈ [0.7 : 0.2 : 1.3]
nbre_img = len(val_b) + len(val_c)          # number of intensity change values ==> number of test images
scale    = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]   # s ∈]1.1 : 0.2 : 2.3]
rot      = [15, 30, 45, 60, 75, 90]         # r ∈ [15 : 15 : 90

DetectorsLegend   = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'fast', 'mser', 'agast', 'gftt', 'gftt_harris', 'star', 'hl', 'msd', 'tbmr']
DescriptorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'daisy', 'freak', 'brief', 'lucid', 'latch', 'vgg', 'beblid', 'teblid', 'boost']
line_styles = ['solid', 'dash', 'dot']
Norm = ['L2', 'HAM']

num_combinations = len(DetectorsLegend) * len(DescriptorsLegend) * len(Norm)
colors = sample_colorscale('Turbo', [i / num_combinations for i in range(num_combinations)])

### detectors/descriptors 5
sift  = cv2.SIFT_create(nfeatures=30000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10.0, sigma=1.6)
akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2, max_points=-1)
orb   = cv2.ORB_create(nfeatures=30000, scaleFactor=1.2, nlevels=6, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
brisk = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
kaze  = cv2.KAZE_create(extended=False, upright=False, threshold=0.004, nOctaves=3, nOctaveLayers=5, diffusivity=cv2.KAZE_DIFF_PM_G2)

### detectors 9
fast  = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,  threshold=5)
mser  = cv2.MSER_create(delta=5, min_area=30, max_area=14400, max_variation=0.15, min_diversity=0.20, max_evolution=400, area_threshold=1.01, min_margin=0.003, edge_blur_size=5)
agast = cv2.AgastFeatureDetector_create(threshold=20, nonmaxSuppression=True, type=cv2.AGAST_FEATURE_DETECTOR_AGAST_5_8)
gftt  = cv2.GFTTDetector_create(maxCorners=30000, qualityLevel=0.01, minDistance=1.0, blockSize=3, useHarrisDetector=False, k=0.04)
gftt_harris = cv2.GFTTDetector_create(maxCorners=30000, qualityLevel=0.01, minDistance=1.0, blockSize=3, useHarrisDetector=True, k=0.04)
star  = cv2.xfeatures2d.StarDetector_create(maxSize=15, responseThreshold=5, lineThresholdProjected=60, lineThresholdBinarized=30, suppressNonmaxSize=3)
hl    = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=4, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=30000, num_layers=4)
msd   = cv2.xfeatures2d.MSDDetector_create(m_patch_radius=3, m_search_area_radius=5, m_nms_radius=5, m_nms_scale_radius=0, m_th_saliency=100.0, m_kNN=4, m_scale_factor=1.25, m_n_scales=-1, m_compute_orientation=0)
tbmr  = cv2.xfeatures2d.TBMR_create(min_area=30, max_area_relative=0.01, scale_factor=1.25, n_scales=2)

### descriptors 9
vgg    = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=6.25, dsc_normalize=False)
daisy  = cv2.xfeatures2d.DAISY_create(radius=15, q_radius=3, q_theta=8, q_hist=8, norm=cv2.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak  = cv2.xfeatures2d.FREAK_create(orientationNormalized=True, scaleNormalized=False, patternScale=22.0, nOctaves=3)
brief  = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=True)
lucid  = cv2.xfeatures2d.LUCID_create(lucid_kernel=3, blur_kernel=6)
latch  = cv2.xfeatures2d.LATCH_create(bytes=2, rotationInvariance=True, half_ssd_size=1, sigma=1.4)
beblid = cv2.xfeatures2d.BEBLID_create(scale_factor=6.25, n_bits=100)
teblid = cv2.xfeatures2d.TEBLID_create(scale_factor=6.25, n_bits=102)
boost  = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=6.25)

Detectors   = list([sift, akaze, orb, brisk, kaze, fast, mser, agast, gftt, gftt_harris, star, hl, msd, tbmr])
#                   0     1      2    3      4     5     6     7      8     9            10    11  12   13
Descriptors = list([sift, akaze, orb, brisk, kaze, daisy, freak, brief, lucid, latch, vgg, beblid, teblid, boost]) 
#                   0     1      2    3      4     5      6      7      8      9      10   11      12      13
matching    = list([cv2.NORM_L2, cv2.NORM_HAMMING])
matcher     = 0 # 0: Brute-force matcher, 1: Flann-based matcher
a = 100 #i
b = 100 #j
drawing = False
save = True

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
    match_rate = (len(good_matches) / len(matches) * 100 if len(matches) > 0 else 0)
    return match_rate, good_matches, matches

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
    
    _, mask = cv2.findFundamentalMat(points1, points2, cv2.USAC_MAGSAC)
    inliers = [matches[i] for i in range(len(matches)) if mask[i] == 1]
    inliers.sort(key=lambda x: x.distance)
    inliers_percentage = ((len(inliers) / len(matches)) * 100 if len(matches) > 0 else 0)
    return inliers_percentage, inliers, matches