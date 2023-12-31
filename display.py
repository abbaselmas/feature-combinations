import matplotlib.pyplot as plt
import numpy as np
import os

val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30] I+b
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3]. I*c
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images
scale = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3] # 7 values of the scale change s ∈]1.1 : 0.2 : 2.3].
rot = [10, 20, 30, 40, 50, 60, 70, 80, 90] # 9 values of rotation change, rotations from 10 to 90 with a step of 10.

# import saved numpy arrays
basedir = os.path.abspath(os.path.dirname(__file__))
Rate_intensity2 = np.load(basedir + '/arrays/Rate_intensity2.npy')
Rate_scale2     = np.load(basedir + '/arrays/Rate_scale2.npy')
Rate_rot2       = np.load(basedir + '/arrays/Rate_rot2.npy')

# Binary and non-binary methods used to set the legend
DetectorsLegend =   ['sift-', 'akaze-', 'orb-', 'brisk-', 'kaze-', 'fast-', 'star-', 'mser-', 'agast-', 'gftt-', 'harrislaplace-', 'msd-',   'tbmr-']
DescriptorsLegend = ['sift',  'akaze',  'orb',  'brisk',  'kaze',  'vgg',   'daisy', 'freak', 'brief',  'lucid', 'latch',          'beblid', 'teblid', 'boost']

c3 = 1 # for binary methods "Detectors with Descriptors"    (c3=0 for bf.L1, c3=1 for bf.L2)

# Number of colors to use for all curves
NUM_COLORS = (len(DetectorsLegend)*len(DescriptorsLegend)) # NUM_COLORS = 5+8*9 = 77
LINE_STYLES = ['solid', 'dashed', 'dotted'] #, 'dashdot'] # style of the curve 4 styles
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('gist_rainbow')
num = -1 # for plot

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

# for the plot, I have inserted the following link: https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
# for i in range(len(DetectorsLegend)):
for j in range(len(DescriptorsLegend)):
    i = 3
    # j = 0
    if i == j:
        continue
    Rate2_I1 = Rate_intensity2[:4,c3,i,j]
    Rate2_I2 = Rate_intensity2[4:,c3,i,j]
    Rate2_S  = Rate_scale2[:,c3,i,j]
    Rate2_R  = Rate_rot2[:,c3,i,j]

    lines_I1 = ax1.plot(val_b, Rate2_I1, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of intensity change results (I+b)
    lines_I2 = ax2.plot(val_c, Rate2_I2, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of intensity change results (I*c)
    lines_S  = ax3.plot(scale, Rate2_S,  linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of the results of scale change
    lines_R  = ax4.plot(rot,   Rate2_R,  linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of the results of rotation change

    num += 4 # to take each time the loop turns a different style of curve
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
if c3 == 0:
    ax1.set_title('Scn. #1 Norm L1 for all methods ', fontsize=13)
    ax2.set_title('Scn. #2 Norm L1 for all methods ', fontsize=13)
    ax3.set_title('Scn. #3 Norm L1 for all methods ', fontsize=13)
    ax4.set_title('Scn. #4 Norm L1 for all methods ', fontsize=13)
elif c3 == 1:
    ax1.set_title('Scn. #1 Norm L2 for all methods ', fontsize=13)
    ax2.set_title('Scn. #2 Norm L2 for all methods ', fontsize=13)
    ax3.set_title('Scn. #3 Norm L2 for all methods ', fontsize=13)
    ax4.set_title('Scn. #4 Norm L2 for all methods ', fontsize=13)


ax1.set_xlabel('Intensity changing I+b', fontsize=10) # x-axis title of the figure
ax1.set_xticks(val_b) # to set the values of the x-axis
ax1.set_ylabel('Correctly matched point rates %', fontsize=10) # title of y-axis of the figure
ax1.grid(True, alpha=0.3)
# ax1.legend(loc="best", bbox_to_anchor=(1, -0.07), shadow=True, fancybox=True, fontsize=7)

ax2.set_xlabel('Intensity changing Ixc', fontsize=10) # x-axis title of the figure
ax2.set_xticks(val_c) # to set the values of the x-axis
ax2.set_ylabel('Correctly matched point rates %', fontsize=10) # title of y-axis of the figure
ax2.grid(True, alpha=0.3)
# ax2.legend(loc="best", ncol=11, bbox_to_anchor=(1, -0.07), shadow=True, fancybox=True, fontsize=7)

ax3.set_xlabel('Scale changing', fontsize=10) # x-axis title of the figure
ax3.set_xticks(scale) # to set the values of the x-axis
ax3.set_ylabel('Correctly matched point rates %', fontsize=10) # title of y-axis of the figure
ax3.grid(True, alpha=0.3)
# ax3.legend(loc="best", ncol=11, bbox_to_anchor=(1, -0.07), shadow=True, fancybox=True, fontsize=7)

ax4.set_xlabel('Rotation changing', fontsize=10) # x-axis title of the figure
ax4.set_xticks(rot) # to set the values of the x-axis
ax4.set_ylabel('Correctly matched point rates %', fontsize=10) # title of y-axis of the figure
ax4.grid(True, alpha=0.3)
# ax4.legend(loc="best", ncol=11, bbox_to_anchor=(1, -0.07), shadow=True, fancybox=True, fontsize=7)

handles, labels = ax1.get_legend_handles_labels() # return lines and labels
# handles, labels = ax2.get_legend_handles_labels()
# handles, labels = ax3.get_legend_handles_labels()
# handles, labels = ax4.get_legend_handles_labels()

fig.legend(handles, labels, loc='outside right center', ncol=3, shadow=True, fancybox=True, fontsize=7)
fig.subplots_adjust(left=0.03, right=0.9, top=0.97, bottom=0.06, hspace=0.2, wspace=0.1)
plt.show()
