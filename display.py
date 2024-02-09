import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import os

val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c)
scale = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # s ∈ [0.5 : 0.2 : 1.5]
rot = [15, 30, 45, 60, 75, 90] # r ∈ [15 : 15 : 90

basedir = os.path.abspath(os.path.dirname(__file__))
Rate_intensity = np.load(basedir + '/arrays/Rate_intensity.npy')
Rate_scale = np.load(basedir + '/arrays/Rate_scale.npy')
Rate_rot = np.load(basedir + '/arrays/Rate_rot.npy')

DetectorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'fast', 'mser', 'agast', 'gftt', 'star', 'hl', 'msd', 'tbmr']
DescriptorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'vgg', 'daisy', 'freak', 'brief', 'lucid', 'latch', 'beblid', 'teblid', 'boost']
line_styles = ['solid', 'dash', 'dot']  # Add more styles as needed

#Norm = ['L1', 'L2', 'L2SQR', 'HAM']
Norm = ['L2', 'HAM']

fig = make_subplots(rows=2, cols=2, subplot_titles=['Scn. #1', 'Scn. #2', 'Scn. #3', 'Scn. #4'], shared_xaxes=False, shared_yaxes=False)

for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate2_I1 = Rate_intensity[:len(val_b), c3, i, j]
            Rate2_I2 = Rate_intensity[len(val_c):, c3, i, j]
            Rate2_S = Rate_scale[:, c3, i, j]
            Rate2_R = Rate_rot[:, c3, i, j]

            color = f'rgba({i * 30}, {j * 20}, {(i + j) * 2}, 1)'  # Adjust as needed
            style = line_styles[j % len(line_styles)]  # Cycle through line styles

            legend_group = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'  # Unique legend group for each trace
            trace_I1 = go.Scatter(x=val_b, y=Rate2_I1, mode='lines', line=dict(color=color, dash=style), name=f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}', legendgroup=legend_group, showlegend= True)
            trace_I2 = go.Scatter(x=val_c, y=Rate2_I2, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
            trace_S  = go.Scatter(x=scale, y=Rate2_S,  mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
            trace_R  = go.Scatter(x=rot,   y=Rate2_R,  mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)

            fig.add_trace(trace_I1, row=1, col=1)
            fig.add_trace(trace_I2, row=1, col=2)
            fig.add_trace(trace_S,  row=2, col=1)
            fig.add_trace(trace_R,  row=2, col=2)

fig.update_layout(  xaxis = dict(tickvals = val_b),
                    xaxis2 = dict(tickvals = val_c),
                    xaxis3 = dict(tickvals = scale),
                    xaxis4 = dict(tickvals = rot))

fig.update_xaxes(title_text="Intensity changing I+b", row=1, col=1)
fig.update_xaxes(title_text="Intensity changing Ixc", row=1, col=2)
fig.update_xaxes(title_text="Scale changing", row=2, col=1)
fig.update_xaxes(title_text="Rotation changing", row=2, col=2)
fig.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)

fig.write_html("PhD.html")
