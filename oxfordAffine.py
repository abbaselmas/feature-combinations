import cv2
import numpy as np
import time, os, csv

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
    match_rate = (len(good_matches) / len(matches) * 100 if len(matches) > 0 else 0)
    return match_rate, good_matches, matches
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
    inliers.sort(key=lambda x: x.distance)
    inliers_percentage = ((len(inliers) / len(matches)) * 100 if len(matches) > 0 else 0)
    return inliers_percentage, inliers, matches
# ................................................................................

### detectors/descriptors 5
sift   = cv2.SIFT_create(nfeatures=2000, nOctaveLayers=3, contrastThreshold=0.1, edgeThreshold=10.0, sigma=1.6)
akaze  = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.01, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)
orb    = cv2.ORB_create(nfeatures=2000, scaleFactor=1.1, nlevels=6, edgeThreshold=60, firstLevel=1, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=60, fastThreshold=60)
brisk  = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
kaze   = cv2.KAZE_create(extended=False, upright=False, threshold=0.01, nOctaves=4, nOctaveLayers=4, diffusivity=cv2.KAZE_DIFF_PM_G2)

### detectors 9
fast    = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,  threshold=5)
mser  = cv2.MSER_create(delta=5, min_area=60, max_area=14400, max_variation=0.25, min_diversity=0.90, max_evolution=20, area_threshold=1.01, min_margin=0.003, edge_blur_size=5)
agast   = cv2.AgastFeatureDetector_create(threshold=20, nonmaxSuppression=True, type=cv2.AGAST_FEATURE_DETECTOR_AGAST_5_8)
gftt  = cv2.GFTTDetector_create(qualityLevel=0.5, minDistance=20.0, blockSize=3, useHarrisDetector=False, k=0.04, maxCorners=2000)
gftt_harris = cv2.GFTTDetector_create(qualityLevel=0.5, minDistance=20.0, blockSize=3, useHarrisDetector=True, k=0.04, maxCorners=2000) 
star  = cv2.xfeatures2d.StarDetector_create(maxSize=20, responseThreshold=5, lineThresholdProjected=100, lineThresholdBinarized=30, suppressNonmaxSize=3)
hl    = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(numOctaves=4, corn_thresh=0.01, DOG_thresh=0.01, maxCorners=2000, num_layers=4)
msd   = cv2.xfeatures2d.MSDDetector_create(m_patch_radius=3, m_search_area_radius=5, m_nms_radius=5, m_nms_scale_radius=0, m_th_saliency=250.0, m_kNN=4, m_scale_factor=1.25, m_n_scales=-1, m_compute_orientation=0)
tbmr  = cv2.xfeatures2d.TBMR_create(min_area=40, max_area_relative=0.01, scale_factor=1.25, n_scales=-1)

### descriptors 9
vgg = cv2.xfeatures2d.VGG_create(desc=103 ,isigma=1.4, img_normalize=False, use_scale_orientation=True, scale_factor=6.25, dsc_normalize=False)
daisy = cv2.xfeatures2d.DAISY_create(radius=15, q_radius=3, q_theta=8, q_hist=8, norm=cv2.xfeatures2d.DAISY_NRM_NONE, interpolation=True, use_orientation=False)
freak = cv2.xfeatures2d.FREAK_create(orientationNormalized=True, scaleNormalized=False, patternScale=22.0, nOctaves=3)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=True)
lucid = cv2.xfeatures2d.LUCID_create(lucid_kernel=3, blur_kernel=6)
latch = cv2.xfeatures2d.LATCH_create(bytes=2, rotationInvariance=True, half_ssd_size=1, sigma=1.4)
beblid = cv2.xfeatures2d.BEBLID_create(scale_factor=6.25, n_bits=100)
teblid = cv2.xfeatures2d.TEBLID_create(scale_factor=6.25, n_bits=102)
boost = cv2.xfeatures2d.BoostDesc_create(desc=100, use_scale_orientation=True, scale_factor=6.25)

Detectors      = list([sift, akaze, orb, brisk, kaze, fast, mser, agast, gftt, gftt_harris, star, hl, msd, tbmr])
#                      0     1      2    3      4     5     6     7      8      9          10    11   12    13
Descriptors    = list([sift, akaze, orb, brisk, kaze, daisy, freak, brief, lucid, latch, vgg, beblid, teblid, boost]) 
#                      0     1      2    3      4     5      6      7      8      9     10    11      12      13
matching       = list([cv2.NORM_L2, cv2.NORM_HAMMING])
matcher        = 0 # 0: Brute-force matcher, 1: Flann-based matcher
a = 100 #i
b = 100 #j
drawing = True

########################################################
def executeScenarios(folder):
    print(time.ctime())
    print(f"Folder: {folder}")
    if a == 100 and b == 100:
        Rate      = np.zeros((6, len(matching), len(Detectors), len(Descriptors), 14))
        Exec_time = np.zeros((6, len(matching), len(Detectors), len(Descriptors), 3))
    else:
        Rate      = np.load(f"{maindir}/arrays/Rate_{folder}.npy")      if os.path.exists(f"{maindir}/arrays/Rate_{folder}.npy")      else np.zeros((6, len(matching), len(Detectors), len(Descriptors), 14))
        Exec_time = np.load(f"{maindir}/arrays/Exec_time_{folder}.npy") if os.path.exists(f"{maindir}/arrays/Exec_time_{folder}.npy") else np.zeros((6, len(matching), len(Detectors), len(Descriptors), 3))
    
    img = [cv2.imread(f"{datasetdir}/{folder}/img{i}.jpg") for i in range(1, 7)]
    keypoints_cache   = np.empty((6, len(Detectors), 2), dtype=object)
    descriptors_cache = np.empty((6, len(Detectors), len(Descriptors), 2), dtype=object)
    for k in range(len(img)):
        # if drawing:
        #     if k != 3:
        #         continue
        for i in range(len(Detectors)):
            if (i == a or a == 100):
                method_dtect = Detectors[i]
                if keypoints_cache[0, i, 0] is None:
                    keypoints1 = method_dtect.detect(img[0], None)
                    keypoints_cache[0, i, 0] = keypoints1
                else:
                    keypoints1 = keypoints_cache[0, i, 0]   
                if keypoints_cache[k, i, 1] is None:
                    start_time = time.time()
                    keypoints2 = method_dtect.detect(img[k], None)
                    detect_time = time.time() - start_time
                    keypoints_cache[k, i, 1] = keypoints2
                else:
                    keypoints2 = keypoints_cache[k, i, 1]
                for j in range(len(Descriptors)):
                    if j == b or b == 100:
                        method_dscrpt = Descriptors[j]
                        for c3 in range(len(matching)):
                            Exec_time[k, c3, i, j, 0] = detect_time
                            Rate[k, c3, i, j, 0] = k
                            Rate[k, c3, i, j, 1] = i
                            Rate[k, c3, i, j, 6] = j
                            Rate[k, c3, i, j, 9] = matching[c3]
                            Rate[k, c3, i, j,10] = matcher
                            try:
                                if descriptors_cache[0, i, j, 0] is None:
                                    keypoints11, descriptors1 = method_dscrpt.compute(img[0], keypoints1)
                                    descriptors_cache[0, i, j, 0] = descriptors1
                                else:
                                    descriptors1 = descriptors_cache[0, i, j, 0]
                                if descriptors_cache[k, i, j, 1] is None:
                                    start_time = time.time()
                                    keypoints22, descriptors2 = method_dscrpt.compute(img[k], keypoints2)
                                    descript_time = time.time() - start_time
                                    descriptors_cache[k, i, j, 1] = descriptors2
                                else:
                                    descriptors2 = descriptors_cache[k, i, j, 1]
                                Exec_time[k, c3, i, j, 1] = descript_time
                                start_time = time.time()
                                Rate[k, c3, i, j, 13], good_matches, matches = evaluate_with_fundamentalMat_and_XSAC(matcher, keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                                Exec_time[k, c3, i, j, 2] = time.time() - start_time
                                Rate[k, c3, i, j, 2] = len(keypoints1)
                                Rate[k, c3, i, j, 3] = len(keypoints11)
                                Rate[k, c3, i, j, 4] = len(keypoints2)
                                Rate[k, c3, i, j, 5] = len(keypoints22)
                                Rate[k, c3, i, j, 7] = len(descriptors1)
                                Rate[k, c3, i, j, 8] = len(descriptors2)
                                Rate[k, c3, i, j,11] = len(good_matches)
                                Rate[k, c3, i, j,12] = len(matches)
                            except:
                                Exec_time[k, c3, i, j, :] = None
                                Rate[k, c3, i, j, 2] = None
                                Rate[k, c3, i, j, 3] = None
                                Rate[k, c3, i, j, 4] = None
                                Rate[k, c3, i, j, 5] = None
                                Rate[k, c3, i, j, 7] = None
                                Rate[k, c3, i, j, 8] = None
                                Rate[k, c3, i, j,11] = None
                                Rate[k, c3, i, j,12] = None
                                Rate[k, c3, i, j,13] = None
                                continue
                            
                            if drawing and k == 3:
                                keypointImage1 = cv2.drawKeypoints(img[0],          keypoints1,  None, color=(206, 217, 162), flags=0)
                                ImageGT        = cv2.drawKeypoints(keypointImage1,  keypoints11, None, color=( 18, 156, 243), flags=0)
                                keypointImage2 = cv2.drawKeypoints(img[k],          keypoints2,  None, color=(206, 217, 162), flags=0)
                                Image2         = cv2.drawKeypoints(keypointImage2,  keypoints22, None, color=(173,  68, 142), flags=0)
                                img_matches    = cv2.drawMatches(ImageGT, keypoints1, Image2, keypoints2, good_matches[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                text = [
                                    f"Detector:     {method_dtect.getDefaultName().split('.')[-1]}",
                                    f"Keypoint1:    {len(keypoints1) if keypoints1 is not None else 0}",
                                    f"Keypoint11:   {len(keypoints11)}",
                                    f"Keypoint2:    {len(keypoints2) if keypoints2 is not None else 0}",
                                    f"Keypoint22:   {len(keypoints22)}",
                                    f"Time Detect:  {Exec_time[k, c3, i, j, 0]:.4f}",
                                    f"Descriptor:   {method_dscrpt.getDefaultName().split('.')[-1]}",
                                    f"Descriptor1:  {len(descriptors1) if descriptors1 is not None else 0}",
                                    f"Descriptor2:  {len(descriptors2) if descriptors2 is not None else 0}",
                                    f"Time Descrpt: {Exec_time[k, c3, i, j, 1]:.4f}",
                                    f"Matching:     {'L2'if matching[c3] == cv2.NORM_L2 else 'HAMMING'}",
                                    f"Matcher:      {'Brute-force' if matcher == 0 else 'Flann-based'}",
                                    f"Match Rate:   {Rate[k, c3, i, j, 13]:.2f}",
                                    f"Time Match:   {Exec_time[k, c3, i, j, 2]:.4f}",
                                    f"Inliers:      {len(good_matches)}",
                                    f"All Matches:  {len(matches)}"
                                ]                                
                                for idx, txt in enumerate(text):
                                    cv2.putText(img_matches, txt, (30, 30+idx*22), cv2.FONT_HERSHEY_COMPLEX , 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                    cv2.putText(img_matches, txt, (30, 30+idx*22), cv2.FONT_HERSHEY_COMPLEX , 0.6, (  0,   0,   0), 1, cv2.LINE_AA)
                                    
                                filename = f"{maindir}/draws/{folder}/{k}_{method_dtect.getDefaultName().split('.')[-1]}_{method_dscrpt.getDefaultName().split('.')[-1]}_{matching[c3]}.png"
                                cv2.imwrite(filename, img_matches)
                    else:
                        continue
            else:
                continue
    np.save(f"{maindir}/arrays/Rate_{folder}.npy",      Rate)
    np.save(f"{maindir}/arrays/Exec_time_{folder}.npy", Exec_time)
    
    headers = [
        "K", "Detector", "Keypoint1", "Keypoint11", "Keypoint2", "Keypoint22",
        "Descriptor", "Descriptor1", "Descriptor2", "Matching", "Matcher",
        "Inliers", "Total Matches", "Match Rate",
        "Detect time", "Descript time", "Match time"
    ]

    with open(f'./csv/{folder}_analysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(headers)
        for k in range(Rate.shape[0]):
            for c3 in range(Rate.shape[1]):
                for i in range(Rate.shape[2]):
                    for j in range(Rate.shape[3]):
                        row = np.append(Rate[k, c3, i, j, :], Exec_time[k, c3, i, j, :])
                        writer.writerow(row)
########################################################
executeScenarios("graf")
executeScenarios("bikes")
executeScenarios("boat")
executeScenarios("leuven")

executeScenarios("wall")
executeScenarios("trees")
executeScenarios("bark")
executeScenarios("ubc")
########################################################
print(time.ctime())