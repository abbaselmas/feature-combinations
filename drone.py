import cv2
import numpy as np
import time, os, csv
from define import *

########################################################
def executeDroneScenarios(a=100, b=100, save=True, drawing=False, matcher=0):
    print(time.ctime())
    print(f"Folder: drone")
    img = [cv2.imread(f"./Small_Buildings/droneResized/DSC00{i}.JPG") for i in range(153, 189)]
    Rate      = np.load(f"./arrays/Rate_drone.npy")      if os.path.exists(f"./arrays/Rate_drone.npy")      else np.zeros((len(img), len(matching), len(Detectors), len(Descriptors), 14))
    Exec_time = np.load(f"./arrays/Exec_time_drone.npy") if os.path.exists(f"./arrays/Exec_time_drone.npy") else np.zeros((len(img), len(matching), len(Detectors), len(Descriptors), 3))
    keypoints_cache   = np.empty((len(img), len(Detectors), 2), dtype=object)
    descriptors_cache = np.empty((len(img), len(Detectors), len(Descriptors), 2), dtype=object)
    for k in range(len(img)-1):
        if drawing:
            if k != 3:
                continue
        for i in range(len(Detectors)):
            if (i == a or a == 100):
                method_dtect = Detectors[i]
                if keypoints_cache[k, i, 0] is None:
                    keypoints1 = method_dtect.detect(img[k], None)
                    keypoints_cache[k, i, 0] = keypoints1
                else:
                    keypoints1 = keypoints_cache[k, i, 0]
                if keypoints_cache[k+1, i, 1] is None:
                    start_time = time.time()
                    keypoints2 = method_dtect.detect(img[k+1], None)
                    detect_time = time.time() - start_time
                    keypoints_cache[k+1, i, 1] = keypoints2
                else:
                    keypoints2 = keypoints_cache[k+1, i, 1]
                for j in range(len(Descriptors)):
                    if j == b or b == 100:
                        method_dscrpt = Descriptors[j]
                        for c3 in range(len(matching)):
                            print(f"Image: {k} Detector: {method_dtect.getDefaultName().split('.')[-1]} Descriptor: {method_dscrpt.getDefaultName().split('.')[-1]} Matching: {matching[c3]}")
                            Exec_time[k, c3, i, j, 0] = detect_time
                            Rate[k, c3, i, j, 0] = k
                            Rate[k, c3, i, j, 1] = i
                            Rate[k, c3, i, j, 2] = j
                            Rate[k, c3, i, j, 3] = matching[c3]
                            Rate[k, c3, i, j, 4] = matcher
                            try:
                                if descriptors_cache[k, i, j, 0] is None:
                                    _, descriptors1 = method_dscrpt.compute(img[k], keypoints1)
                                    descriptors_cache[k, i, j, 0] = descriptors1
                                else:
                                    descriptors1 = descriptors_cache[k, i, j, 0]
                                if descriptors_cache[k+1, i, j, 1] is None:
                                    start_time = time.time()
                                    _, descriptors2 = method_dscrpt.compute(img[k+1], keypoints2)
                                    descript_time = time.time() - start_time
                                    descriptors_cache[k+1, i, j, 1] = descriptors2
                                else:
                                    descriptors2 = descriptors_cache[k+1, i, j, 1]
                                Exec_time[k, c3, i, j, 1] = descript_time
                                start_time = time.time()
                                Rate[k, c3, i, j, 11], good_matches, matches, h = evaluate_with_fundamentalMat_and_XSAC(matcher, keypoints1, keypoints2, descriptors1, descriptors2, matching[c3])
                                Exec_time[k, c3, i, j, 2] = time.time() - start_time
                                Rate[k, c3, i, j, 5] = len(keypoints1)
                                Rate[k, c3, i, j, 6] = len(keypoints2)
                                Rate[k, c3, i, j, 7] = len(descriptors1)
                                Rate[k, c3, i, j, 8] = len(descriptors2)
                                Rate[k, c3, i, j, 9] = len(good_matches)
                                Rate[k, c3, i, j,10] = len(matches)
                            except:
                                Exec_time[k, c3, i, j, :] = None
                                Rate[k, c3, i, j, 5] = None
                                Rate[k, c3, i, j, 6] = None
                                Rate[k, c3, i, j, 7] = None
                                Rate[k, c3, i, j, 8] = None
                                Rate[k, c3, i, j, 9] = None
                                Rate[k, c3, i, j,10] = None
                                Rate[k, c3, i, j,11] = None
                                continue
                            
                            if drawing and k == 3 and Rate[k, c3, i, j, 9] > 1000:
                                img_matches    = cv2.drawMatches(img[k], keypoints1, img[k+1], keypoints2, good_matches[:], None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                                text = [
                                    f"Detector:     {method_dtect.getDefaultName().split('.')[-1]}",
                                    f"Keypoint1:    {len(keypoints1) if keypoints1 is not None else 0}",
                                    f"Keypoint2:    {len(keypoints2) if keypoints2 is not None else 0}",
                                    f"Time Detect:  {Exec_time[k, c3, i, j, 0]:.4f}",
                                    f"Descriptor:   {method_dscrpt.getDefaultName().split('.')[-1]}",
                                    f"Descriptor1:  {len(descriptors1) if descriptors1 is not None else 0}",
                                    f"Descriptor2:  {len(descriptors2) if descriptors2 is not None else 0}",
                                    f"Time Descrpt: {Exec_time[k, c3, i, j, 1]:.4f}",
                                    f"Matching:     {'L2'if matching[c3] == cv2.NORM_L2 else 'HAMMING'}",
                                    f"Matcher:      {'Brute-force' if matcher == 0 else 'Flann-based'}",
                                    f"Match Rate:   {Rate[k, c3, i, j, 11]:.2f}",
                                    f"Time Match:   {Exec_time[k, c3, i, j, 2]:.4f}",
                                    f"Inliers:      {len(good_matches)}",
                                    f"All Matches:  {len(matches)}"
                                ]                                
                                for idx, txt in enumerate(text):
                                    cv2.putText(img_matches, txt, (30, 30+idx*22), cv2.FONT_HERSHEY_COMPLEX , 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                    cv2.putText(img_matches, txt, (30, 30+idx*22), cv2.FONT_HERSHEY_COMPLEX , 0.6, (  0,   0,   0), 1, cv2.LINE_AA)
                                    
                                filename = f"./draws/drone/{k}_{method_dtect.getDefaultName().split('.')[-1]}_{method_dscrpt.getDefaultName().split('.')[-1]}_{matching[c3]}.png"
                                cv2.imwrite(filename, img_matches)
                    else:
                        continue
            else:
                continue
    if save:
        np.save(f"./arrays/Rate_drone.npy",      Rate)
        np.save(f"./arrays/Exec_time_drone.npy", Exec_time)
        headers = ["K", "Detector", "Descriptor", "Matching", "Matcher", "Keypoint1", "Keypoint2", "Descriptor1", "Descriptor2", "Inliers", "Total Matches", "Match Rate", "Detect time", "Descript time", "Match time"]
        with open(f'./csv/drone_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(headers)
            for k in range(Rate.shape[0]):
                for c3 in range(Rate.shape[1]):
                    for i in range(Rate.shape[2]):
                        for j in range(Rate.shape[3]):
                            row = np.append(Rate[k, c3, i, j, :], Exec_time[k, c3, i, j, :])
                            writer.writerow(row)

################MAIN CODE###################################
executeDroneScenarios(a=100, b=100, save=True, drawing=False, matcher=0)
print(time.ctime())
########################################################