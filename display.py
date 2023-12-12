import matplotlib.pyplot as plt
import numpy as np
import os

val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c) # number of intensity change values ==> number of test images
scale = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3] # 7 values of the scale change s ∈]1.1 : 0.2 : 2.3].
rot = [10, 20, 30, 40, 50, 60, 70, 80, 90] # 9 values of rotation change, rotations from 10 to 90 with a step of 10.

# import saved numpy arrays
basedir = os.path.abspath(os.path.dirname(__file__))
Rate_intensity1 = np.load(basedir + '/arrays/Rate_intensity1.npy')
Rate_intensity2 = np.load(basedir + '/arrays/Rate_intensity2.npy')
Rate_scale1     = np.load(basedir + '/arrays/Rate_scale1.npy')
Rate_scale2     = np.load(basedir + '/arrays/Rate_scale2.npy')
Rate_rot1       = np.load(basedir + '/arrays/Rate_rot1.npy')
Rate_rot2       = np.load(basedir + '/arrays/Rate_rot2.npy')

# ...................................................................................................................
# I.3 Display of results
# ...................................................................................................................

# Binary and non-binary methods used to set the legend
DetectDescriptLegend = ['sift',  'akaze', 'orb',   'brisk',  'kaze']
DetectorsLegend      = ['fast-', 'star-', 'mser-', 'agast-', 'gftt-', 'harrislaplace-', 'msd-',   'tbmr-']
DescriptorsLegend    = ['vgg',   'daisy', 'freak', 'brief',  'lucid', 'latch',          'beblid', 'teblid', 'boost']

c2 = 1 # for non-binary methods "DetectDescript"            (c2=0 for bf.L1, c2=1 for bf.L2)
c3 = 1 # for binary methods "Detectors with Descriptors"    (c3=0 for bf.L1, c3=1 for bf.L2)

# Number of colors to use for all curves
NUM_COLORS = len(DetectDescriptLegend) + (len(DetectorsLegend)*len(DescriptorsLegend)) # NUM_COLORS = 5+8+9 = 22

LINE_STYLES = ['solid', 'dashed', 'dotted'] # style of the curve
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('gist_rainbow')
num = -1 # for plot
# Initialization of the 4 figures
fig1 = plt.figure(1,figsize= (15,10))
fig2 = plt.figure(2,figsize= (15,10))
fig3 = plt.figure(3,figsize= (15,10))
fig4 = plt.figure(4,figsize= (15,10))
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)

# for the plot, I have inserted the following link: https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
# for loop to display the results of non-binary methods
for k in range(len(DetectDescriptLegend)):
    Rate1_I1 = Rate_intensity1[:4, c2, k]
    Rate1_I2 = Rate_intensity1[4:, c2, k]
    Rate1_S  = Rate_scale1[:, c2, k]
    Rate1_R  = Rate_rot1[:, c2, k]

    lines_I1 = ax1.plot(val_b, Rate1_I1, linewidth=2, label = DetectDescriptLegend[k]) # for the figure of the intensity change results (I+b)
    lines_I2 = ax2.plot(val_c, Rate1_I2, linewidth=2, label = DetectDescriptLegend[k]) # for the figure of intensity change results (I*c)
    lines_S  = ax3.plot(scale, Rate1_S,  linewidth=2, label = DetectDescriptLegend[k]) # for the scaling results figure
    lines_R  = ax4.plot(rot, Rate1_R,    linewidth=2, label = DetectDescriptLegend[k]) # for the figure of the results of rotation change

    num += 1 # to take each time the loop turns a different color and curve style
    # for the color and style of the curve for the results of the 3 scenarios
    lines_I1[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_I1[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_I2[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_I2[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_S[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_S[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])
    lines_R[0].set_color(cm(num//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines_R[0].set_linestyle(LINE_STYLES[num%NUM_STYLES])

# for loop to display the results of binary methods
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        Rate2_I1 = Rate_intensity2[:4,c3,i,j]
        Rate2_I2 = Rate_intensity2[4:,c3,i,j]
        Rate2_S  = Rate_scale2[:,c3,i,j]
        Rate2_R  = Rate_rot2[:,c3,i,j]

        lines_I1 = ax1.plot(val_b, Rate2_I1, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of intensity change results (I+b)
        lines_I2 = ax2.plot(val_c, Rate2_I2, linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of intensity change results (I*c)
        lines_S  = ax3.plot(scale, Rate2_S,  linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of the results of scale change
        lines_R  = ax4.plot(rot,   Rate2_R,  linewidth=2, label = DetectorsLegend[i] + DescriptorsLegend[j]) # for the figure of the results of rotation change

        num += 1 # to take each time the loop turns a different style of curve
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
if c2 == 0 and c3 == 0:
    ax1.set_title('Scn.1 L1 (non-binary) and L1 (binary) meth.', fontsize=13)
    ax2.set_title('Scn.2 L1 (non-binary) and L1 (binary) meth.', fontsize=13)
    ax3.set_title('Scn.3 L1 (non-binary) and L1 (binary) meth.', fontsize=13)
    ax4.set_title('Scn.4 L1 (non-binary) and L1 (binary) meth.', fontsize=13)
elif c2 == 1 and c3 == 1:
    ax1.set_title('Scn.1 L2 (non-binary) and L2 (binary) meth.', fontsize=13)
    ax2.set_title('Scn.2 L2 (non-binary) and L2 (binary) meth.', fontsize=13)
    ax3.set_title('Scn.3 L2 (non-binary) and L2 (binary) meth.', fontsize=13)
    ax4.set_title('Scn.4 L2 (non-binary) and L2 (binary) meth.', fontsize=13)
elif c2 == 0 and c3 == 1:
    ax1.set_title('Scn.1 L1 (non-binary) and L2 (binary) meth.', fontsize=13)
    ax2.set_title('Scn.2 L1 (non-binary) and L2 (binary) meth.', fontsize=13)
    ax3.set_title('Scn.3 L1 (non-binary) and L2 (binary) meth.', fontsize=13)
    ax4.set_title('Scn.4 L1 (non-binary) and L2 (binary) meth.', fontsize=13)
elif c2 == 1 and c3 == 0:
    ax1.set_title('Scn.1 L2 (non-binary) and L1 (binary) meth.', fontsize=13)
    ax2.set_title('Scn.2 L2 (non-binary) and L1 (binary) meth.', fontsize=13)
    ax3.set_title('Scn.3 L2 (non-binary) and L1 (binary) meth.', fontsize=13)
    ax4.set_title('Scn.4 L2 (non-binary) and L1 (binary) meth.', fontsize=13)

ax1.set_xlabel('Intensity changing I+b', fontsize=12) # x-axis title of the figure
ax1.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax1.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize=7, handlelength = 2) # legend :(loc=2 <=> Location String = 'upper left')

# ax2.set_title('Correctly matched point rate for different matching methods depending on intensity change', fontsize=13)
ax2.set_xlabel('Intensity changing Ixc', fontsize=12) # x-axis title of the figure
ax2.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax2.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize=7, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# ax3.set_title('Correctly matched point rate for different matching methods depending on scale change', fontsize=13)
ax3.set_xlabel('Scale changing', fontsize=12) # x-axis title of the figure
ax3.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax3.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize=7, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# ax4.set_title('Correctly matched point rate for different pairing methods depending on the change of rotation', fontsize=13)
ax4.set_xlabel('Rotation changing', fontsize=12) # x-axis title of the figure
ax4.set_ylabel('Correctly matched point rates %', fontsize=12) # title of y-axis of the figure
ax4.legend(loc= 'center left', bbox_to_anchor=(1, 0.5), fontsize=7, handlelength = 2) # (loc=2 <=> Location String = 'upper left')

# Recording and display of the obtained figures
fig1.savefig(basedir + '\\figs\\' + ax1.get_title() + ax1.get_xlabel() + '.png')
fig2.savefig(basedir + '\\figs\\' + ax2.get_title() + ax2.get_xlabel() + '.png')
fig3.savefig(basedir + '\\figs\\' + ax3.get_title() + ax3.get_xlabel() + '.png')
fig4.savefig(basedir + '\\figs\\' + ax4.get_title() + ax4.get_xlabel() + '.png')
plt.show()
