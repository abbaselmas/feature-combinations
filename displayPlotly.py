import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import os

val_b = np.array([-30, -10, 0, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c)
scale = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] # 7 values of the scale change s ∈]1.1 : 0.2 : 2.3].
rot = [-45, -30, -20, -10, 0, 10, 20, 30, 45] # 9 values of rotation change, rotations from 10 to 90 with a step of 10.

# Load data
basedir = os.path.abspath(os.path.dirname(__file__))
Rate_intensity = np.load(basedir + '/arrays/Rate_intensity.npy')
Rate_scale = np.load(basedir + '/arrays/Rate_scale.npy')
Rate_rot = np.load(basedir + '/arrays/Rate_rot.npy')

DetectorsLegend = ['sift-', 'akaze-', 'orb-', 'brisk-', 'kaze-', 'fast-', 'mser-', 'agast-', 'gftt-', 'star-', 'harrislaplace-', 'msd-', 'tbmr-']
DescriptorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'vgg', 'daisy', 'freak', 'brief', 'lucid', 'latch', 'beblid', 'teblid', 'boost']
line_styles = ['solid', 'dash', 'dot']  # Add more styles as needed

c3 = 0

fig = make_subplots(rows=2, cols=2, subplot_titles=['Scn. #1 Norm L2 for all methods', 'Scn. #2 Norm L2 for all methods', 'Scn. #3 Norm L2 for all methods', 'Scn. #4 Norm L2 for all methods'], shared_xaxes=False, shared_yaxes=False)

for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        Rate2_I1 = Rate_intensity[:len(val_b), c3, i, j]
        Rate2_I2 = Rate_intensity[len(val_c):, c3, i, j]
        Rate2_S = Rate_scale[:, c3, i, j]
        Rate2_R = Rate_rot[:, c3, i, j]

        color = f'rgba({i * 30}, {j * 20}, {(i + j) * 2}, 1)'  # Adjust as needed
        style = line_styles[j % len(line_styles)]  # Cycle through line styles

        legend_group = f'{DetectorsLegend[i]}{DescriptorsLegend[j]}'  # Unique legend group for each trace

        trace_I1 = go.Scatter(x=val_b, y=Rate2_I1, mode='lines', line=dict(color=color, dash=style), name=f'{DetectorsLegend[i]}{DescriptorsLegend[j]}', legendgroup=legend_group, showlegend= True)
        trace_I2 = go.Scatter(x=val_c, y=Rate2_I2, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
        trace_S  = go.Scatter(x=scale, y=Rate2_S,  mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
        trace_R  = go.Scatter(x=rot,   y=Rate2_R,  mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)

        fig.add_trace(trace_I1, row=1, col=1)
        fig.add_trace(trace_I2, row=1, col=2)
        fig.add_trace(trace_S,  row=2, col=1)
        fig.add_trace(trace_R,  row=2, col=2)

fig.update_xaxes(title_text="Intensity changing I+b", row=1, col=1)
fig.update_xaxes(title_text="Intensity changing Ixc", row=1, col=2)
fig.update_xaxes(title_text="Scale changing", row=2, col=1)
fig.update_xaxes(title_text="Rotation changing", row=2, col=2)
fig.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)

fig.write_html("PhD.html")