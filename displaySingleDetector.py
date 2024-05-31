import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
import numpy as np
import os

val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
scale = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # s ∈ [0.5 : 0.2 : 1.5]
rot   = [15, 30, 45, 60, 75, 90] # r ∈ [15 : 15 : 90

DetectorsLegend   = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'fast', 'mser', 'agast', 'gftt', 'gftt_harris', 'star', 'hl', 'msd', 'tbmr']
DescriptorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'daisy', 'freak', 'brief', 'lucid', 'latch', 'vgg', 'beblid', 'teblid', 'boost']
line_styles = ['solid', 'dash', 'dot']
Norm = ['L2', 'HAM']

maindir = os.path.abspath(os.path.dirname(__file__))

num_combinations = len(DescriptorsLegend) * len(Norm)
colors = sample_colorscale('Turbo', [i / num_combinations for i in range(num_combinations)])

########################
# MARK: - Synthetic Data
########################
Rate_intensity = np.load(maindir + '/arrays/Rate_intensity.npy')
Rate_scale     = np.load(maindir + '/arrays/Rate_scale.npy')
Rate_rot       = np.load(maindir + '/arrays/Rate_rot.npy')

fig1 = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig1.update_layout(margin=dict(l=20, r=20, t=25, b=25))
fig1.update_layout(xaxis = dict(tickvals = val_b), xaxis2 = dict(tickvals = val_c), xaxis3 = dict(tickvals = scale), xaxis4 = dict(tickvals = rot))
fig1.update_xaxes(title_text="Intensity changing I+b", row=1, col=1)
fig1.update_xaxes(title_text="Intensity changing Ixc", row=1, col=2)
fig1.update_xaxes(title_text="Scale changing", row=2, col=1)
fig1.update_xaxes(title_text="Rotation changing", row=2, col=2)
fig1.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig1.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig1.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig1.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
fig1.update_layout(hovermode="x unified")
for i in range(len(DetectorsLegend)):
    color_index = 0
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate2_I1 = Rate_intensity[:len(val_b), c3, i, j, 11]
            Rate2_I2 = Rate_intensity[len(val_c):, c3, i, j, 11]
            Rate2_S  = Rate_scale    [          :, c3, i, j, 11]
            Rate2_R  = Rate_rot      [          :, c3, i, j, 11]

            color = colors[color_index]
            style = line_styles[j % len(line_styles)]
            legend_groupfig1 = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_intensity[:len(val_b), c3, i, j, 11]).any()):
                fig1trace_I1    = go.Scatter(x=val_b, y=Rate2_I1, mode='lines', line=dict(color=color, dash=style), name=legend_groupfig1, legendgroup=legend_groupfig1, showlegend=True)
                fig1.add_trace(fig1trace_I1, row=1, col=1)
            if not (np.isnan(Rate_intensity[len(val_c):, c3, i, j, 11]).any()):
                fig1trace_I2    = go.Scatter(x=val_c, y=Rate2_I2, mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig1, showlegend=False)
                fig1.add_trace(fig1trace_I2, row=1, col=2)
            if not (np.isnan(Rate_scale[1:, c3, i, j, 11]).any()):
                fig1trace_Scale = go.Scatter(x=scale, y=Rate2_S,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig1, showlegend=False)
                fig1.add_trace(fig1trace_Scale,  row=2, col=1)
            if not (np.isnan(Rate_rot[1:, c3, i, j, 11]).any()):
                fig1trace_Rot   = go.Scatter(x=rot,   y=Rate2_R,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig1, showlegend=False)
                fig1.add_trace(fig1trace_Rot,  row=2, col=2)
            color_index += 1
    fig1.write_html(f'./html/SyntheticData_Detector_{DetectorsLegend[i]}.html')
    fig1.data = []
    fig1trace_I1 = fig1trace_I2 = fig1trace_Scale = fig1trace_Rot = legend_groupfig1 = None
######################
# MARK: - Oxford 1234
######################
Rate_graf   = np.load(maindir + '/arrays/Rate_graf.npy')
Rate_bikes  = np.load(maindir + '/arrays/Rate_bikes.npy')
Rate_boat   = np.load(maindir + '/arrays/Rate_boat.npy')
Rate_leuven = np.load(maindir + '/arrays/Rate_leuven.npy')

fig2 = make_subplots(rows=2, cols=2, subplot_titles=['Graf(Viewpoint)', 'Bikes(Blur)', 'Boat(Zoom + Rotation)', 'Leuven(Light)'], shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig2.update_layout(margin=dict(l=20, r=20, t=25, b=25))
x = ["Img2", "Img3", "Img4", "Img5", "Img6"]
fig2.update_layout(xaxis = dict(tickmode = 'array', tickvals = x), xaxis2 = dict(tickmode = 'array', tickvals = x), xaxis3 = dict(tickmode = 'array', tickvals = x), xaxis4 = dict(tickmode = 'array', tickvals = x))
fig2.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig2.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig2.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig2.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
fig2.update_layout(hovermode="x unified")
for i in range(len(DetectorsLegend)):
    color_index = 0
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate_Graf   = Rate_graf  [1:, c3, i, j, 11]
            Rate_Bikes  = Rate_bikes [1:, c3, i, j, 11]
            Rate_Boat   = Rate_boat  [1:, c3, i, j, 11]
            Rate_Leuven = Rate_leuven[1:, c3, i, j, 11]

            color = colors[color_index]
            style = line_styles[i % len(line_styles)]
            legend_groupfig2 = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_graf[1:, c3, i, j, 11]).any()):
                fig2trace_Graf   = go.Scatter(x=x, y=Rate_Graf,   mode='lines', line=dict(color=color, dash=style), name=legend_groupfig2, legendgroup=legend_groupfig2, showlegend=True)
                fig2.add_trace(fig2trace_Graf, row=1, col=1)
            if not (np.isnan(Rate_bikes[1:, c3, i, j, 11]).any()):
                fig2trace_Bikes  = go.Scatter(x=x, y=Rate_Bikes,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig2, showlegend=False)
                fig2.add_trace(fig2trace_Bikes, row=2, col=2)
            if not (np.isnan(Rate_boat[1:, c3, i, j, 11]).any()):
                fig2trace_Boat   = go.Scatter(x=x, y=Rate_Boat,   mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig2, showlegend=False)
                fig2.add_trace(fig2trace_Boat, row=1, col=2)
            if not (np.isnan(Rate_leuven[1:, c3, i, j, 11]).any()):
                fig2trace_Leuven = go.Scatter(x=x, y=Rate_Leuven, mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig2, showlegend=False)
                fig2.add_trace(fig2trace_Leuven,  row=2, col=1)
            color_index += 1
    fig2.write_html(f'./html/oxfordAffine1234_Detector_{DetectorsLegend[i]}.html')
    fig2.data = []
    fig2trace_Graf = fig2trace_Bikes = fig2trace_Boat  = fig2trace_Leuven  = legend_groupfig2 = None
######################
# MARK: - Oxford 5678
######################
Rate_wall  = np.load(maindir + '/arrays/Rate_wall.npy')
Rate_trees = np.load(maindir + '/arrays/Rate_trees.npy')
Rate_bark  = np.load(maindir + '/arrays/Rate_bark.npy')
Rate_ubc   = np.load(maindir + '/arrays/Rate_ubc.npy')

fig3 = make_subplots(rows=2, cols=2, subplot_titles=['Wall(Viewpoint)', 'Trees(Blur)', 'Bark(Zoom + Rotation)', 'UBC(JPEG)'], shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig3.update_layout(margin=dict(l=20, r=20, t=25, b=25))
x = ["Img2", "Img3", "Img4", "Img5", "Img6"]
fig3.update_layout(xaxis = dict(tickmode = 'array', tickvals = x), xaxis2 = dict(tickmode = 'array', tickvals = x), xaxis3 = dict(tickmode = 'array', tickvals = x), xaxis4 = dict(tickmode = 'array', tickvals = x))
fig3.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig3.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig3.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig3.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
fig3.update_layout(hovermode="x unified")
for i in range(len(DetectorsLegend)):
    color_index = 0
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate_Wall  = Rate_wall [1:, c3, i, j, 11]
            Rate_Trees = Rate_trees[1:, c3, i, j, 11]
            Rate_Bark  = Rate_bark [1:, c3, i, j, 11]
            Rate_Ubc   = Rate_ubc  [1:, c3, i, j, 11]

            color = colors[color_index]
            style = line_styles[i % len(line_styles)]
            legend_groupfig3 = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_bark[1:, c3, i, j, 11]).any()):
                fig3trace_Bark  = go.Scatter(x=x, y=Rate_Bark,  mode='lines', line=dict(color=color, dash=style), name=legend_groupfig3, legendgroup=legend_groupfig3, showlegend=True)
                fig3.add_trace(fig3trace_Bark,  row=1, col=1)
            if not (np.isnan(Rate_wall[1:, c3, i, j, 11]).any()):
                fig3trace_Wall  = go.Scatter(x=x, y=Rate_Wall,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig3, showlegend=False)
                fig3.add_trace(fig3trace_Wall, row=1, col=2)
            if not (np.isnan(Rate_trees[1:, c3, i, j, 11]).any()):
                fig3trace_Trees = go.Scatter(x=x, y=Rate_Trees, mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig3, showlegend=False)
                fig3.add_trace(fig3trace_Trees, row=2, col=1)
            if not (np.isnan(Rate_ubc[1:, c3, i, j, 11]).any()):
                fig3trace_Ubc   = go.Scatter(x=x, y=Rate_Ubc,   mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig3, showlegend=False)
                fig3.add_trace(fig3trace_Ubc,  row=2, col=2)
            color_index += 1
    fig3.write_html(f'./html/oxfordAffine5678_Detector_{DetectorsLegend[i]}.html')
    fig3.data = []
    fig3trace_Bark = fig3trace_Wall = fig3trace_Trees = fig3trace_Ubc = legend_groupfig3 = None
##############################################################################################################