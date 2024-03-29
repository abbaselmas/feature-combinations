import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import os

val_b = np.array([-30, -10, 10, 30]) # b ∈ [−30 : 20 : +30]
val_c = np.array([0.7, 0.9, 1.1, 1.3]) # c ∈ [0.7 : 0.2 : 1.3].
nbre_img = len(val_b) + len(val_c)
scale = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5] # s ∈ [0.5 : 0.2 : 1.5]
rot = [15, 30, 45, 60, 75, 90] # r ∈ [15 : 15 : 90

DetectorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'fast', 'mser', 'agast', 'gftt', 'gftt_harris', 'star', 'hl', 'msd', 'tbmr']
DescriptorsLegend = ['sift', 'akaze', 'orb', 'brisk', 'kaze', 'vgg', 'daisy', 'freak', 'brief', 'lucid', 'latch', 'beblid', 'teblid', 'boost']
line_styles = ['solid', 'dash', 'dot']
Norm = ['L2', 'HAM']

maindir = os.path.abspath(os.path.dirname(__file__))

"""
..######..##....##.##....##.########.##.....##.########.########.####..######.....########.....###....########....###...
.##....##..##..##..###...##....##....##.....##.##..........##.....##..##....##....##.....##...##.##......##......##.##..
.##.........####...####..##....##....##.....##.##..........##.....##..##..........##.....##..##...##.....##.....##...##.
..######.....##....##.##.##....##....#########.######......##.....##..##..........##.....##.##.....##....##....##.....##
.......##....##....##..####....##....##.....##.##..........##.....##..##..........##.....##.#########....##....#########
.##....##....##....##...###....##....##.....##.##..........##.....##..##....##....##.....##.##.....##....##....##.....##
..######.....##....##....##....##....##.....##.########....##....####..######.....########..##.....##....##....##.....##
"""
Rate_intensity  = np.load(maindir + '/arrays/Rate_intensity.npy')
Rate_scale      = np.load(maindir + '/arrays/Rate_scale.npy')
Rate_rot        = np.load(maindir + '/arrays/Rate_rot.npy')

fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig.update_layout(margin=dict(l=20, r=20, t=25, b=25))
fig.update_layout(xaxis = dict(tickvals = val_b), xaxis2 = dict(tickvals = val_c), xaxis3 = dict(tickvals = scale), xaxis4 = dict(tickvals = rot))
fig.update_xaxes(title_text="Intensity changing I+b", row=1, col=1)
fig.update_xaxes(title_text="Intensity changing Ixc", row=1, col=2)
fig.update_xaxes(title_text="Scale changing", row=2, col=1)
fig.update_xaxes(title_text="Rotation changing", row=2, col=2)
fig.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate2_I1 = Rate_intensity[:len(val_b), c3, i, j]
            Rate2_I2 = Rate_intensity[len(val_c):, c3, i, j]
            Rate2_S  = Rate_scale[:, c3, i, j]
            Rate2_R  = Rate_rot[:, c3, i, j]

            color = f'rgba({i * 30}, {j * 20}, {(i + j) * 2}, 1)'
            style = line_styles[j % len(line_styles)]

            legend_group = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_intensity[:len(val_b), c3, i, j]).any() or np.all(Rate_intensity[:len(val_b), c3, i, j]==0)):
                trace_I1 = go.Scatter(x=val_b, y=Rate2_I1, mode='lines', line=dict(color=color, dash=style), name=legend_group, legendgroup=legend_group, showlegend=True)
                fig.add_trace(trace_I1, row=1, col=1)
            if not (np.isnan(Rate_intensity[len(val_c):, c3, i, j]).any() or np.all(Rate_intensity[len(val_c):, c3, i, j]==0)):    
                trace_I2 = go.Scatter(x=val_c, y=Rate2_I2, mode='lines', line=dict(color=color, dash=style), name='',           legendgroup=legend_group, showlegend=False)
                fig.add_trace(trace_I2, row=1, col=2)
            if not (np.isnan(Rate_scale[:, c3, i, j]).any()               or np.all(Rate_scale[:, c3, i, j]==0)):               
                trace_S  = go.Scatter(x=scale, y=Rate2_S,  mode='lines', line=dict(color=color, dash=style), name='',           legendgroup=legend_group, showlegend=False)
                fig.add_trace(trace_S,  row=2, col=1)
            if not (np.isnan(Rate_rot[:, c3, i, j]).any()                 or np.all(Rate_rot[:, c3, i, j]==0)):
                trace_R  = go.Scatter(x=rot,   y=Rate2_R,  mode='lines', line=dict(color=color, dash=style), name='',           legendgroup=legend_group, showlegend=False)
                fig.add_trace(trace_R,  row=2, col=2)               

fig.write_html("./html/SyntheticData.html")
###########################################################################################################

"""
..######..##....##.##....##.########.##.....##....########.####.##.....##.####.##....##..######..
.##....##..##..##..###...##....##....##.....##.......##.....##..###...###..##..###...##.##....##.
.##.........####...####..##....##....##.....##.......##.....##..####.####..##..####..##.##.......
..######.....##....##.##.##....##....#########.......##.....##..##.###.##..##..##.##.##.##...####
.......##....##....##..####....##....##.....##.......##.....##..##.....##..##..##..####.##....##.
.##....##....##....##...###....##....##.....##.......##.....##..##.....##..##..##...###.##....##.
..######.....##....##....##....##....##.....##.......##....####.##.....##.####.##....##..######..
"""
######################################################################################
Exec_time_intensity = np.load(maindir + '/arrays/Exec_time_intensity.npy')
Exec_time_scale     = np.load(maindir + '/arrays/Exec_time_scale.npy')
Exec_time_rot       = np.load(maindir + '/arrays/Exec_time_rot.npy')

fig1 = make_subplots(rows=2, cols=2, subplot_titles=['Detectors', 'Descriptors', 'Evaluation(matching)'], shared_xaxes=False, shared_yaxes=False, specs=[[{}, {}],[{"colspan": 2}, None]],horizontal_spacing=0.05, vertical_spacing=0.1)
fig1.update_layout(margin=dict(l=20, r=20, t=25, b=25))
fig1.update_yaxes(title_text="milliseconds", row=1, col=1)
fig1.update_yaxes(title_text="milliseconds", row=1, col=2)
fig1.update_yaxes(title_text="milliseconds", row=2, col=1)
for i in range(len(DetectorsLegend)):
    mean_detector_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, i, :, 0], Exec_time_scale[:, :, i, :, 0], Exec_time_rot[:, :, i, :, 0]), axis=0)))
    trace_detect = go.Bar(x=[DetectorsLegend[i]], y=[mean_detector_time], name=DetectorsLegend[i], showlegend=True, text=[f'{mean_detector_time:.4f}'], textposition='auto')
    fig1.add_trace(trace_detect, row=1, col=1)
            
for j in range(len(DescriptorsLegend)):
    mean_descriptor_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, :, j, 1], Exec_time_scale[:, :, :, j, 1], Exec_time_rot[:, :, :, j, 1]), axis=0)))
    trace_descr = go.Bar(x=[DescriptorsLegend[j]], y=[mean_descriptor_time], name=DescriptorsLegend[j], showlegend=True, text=[f'{mean_descriptor_time:.4f}'], textposition='auto')
    fig1.add_trace(trace_descr, row=1, col=2)

for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        mean_matching_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, i, j, 2], Exec_time_scale[:, :, i, j, 2], Exec_time_rot[:, :, i, j, 2]), axis=0)))
        if not (np.isnan(mean_matching_time) or mean_matching_time <= 0):
            trace_match = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[mean_matching_time], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{mean_matching_time:.4f}'], textposition='auto')
            fig1.add_trace(trace_match, row=2, col=1)

fig1.write_html("./html/SyntheticData_timing.html")

###########################################################################################################
"""
..#######..##.....##.########..#######..########..########........##....#######...#######..##.......
.##.....##..##...##..##.......##.....##.##.....##.##.....##.....####...##.....##.##.....##.##....##.
.##.....##...##.##...##.......##.....##.##.....##.##.....##.......##..........##........##.##....##.
.##.....##....###....######...##.....##.########..##.....##.......##....#######...#######..##....##.
.##.....##...##.##...##.......##.....##.##...##...##.....##.......##...##...............##.#########
.##.....##..##...##..##.......##.....##.##....##..##.....##.......##...##........##.....##.......##.
..#######..##.....##.##........#######..##.....##.########......######.#########..#######........##.
"""
Rate_graf  = np.load(maindir + '/arrays/Rate_graf.npy')
Rate_wall  = np.load(maindir + '/arrays/Rate_wall.npy')
Rate_trees = np.load(maindir + '/arrays/Rate_trees.npy')
Rate_bikes = np.load(maindir + '/arrays/Rate_bikes.npy')

fig2 = make_subplots(rows=2, cols=2, subplot_titles=['Graf(Viewpoint)', 'Wall(Viewpoint)', 'Trees(Blur)', 'Bikes(Blur)'], shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig2.update_layout(margin=dict(l=20, r=20, t=25, b=25))
x = ["Img2", "Img3", "Img4", "Img5", "Img6"]
fig2.update_layout(xaxis = dict(tickmode = 'array', tickvals = x), xaxis2 = dict(tickmode = 'array', tickvals = x), xaxis3 = dict(tickmode = 'array', tickvals = x), xaxis4 = dict(tickmode = 'array', tickvals = x))
fig2.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig2.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig2.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig2.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate_G = Rate_graf [1:, c3, i, j]
            Rate_W = Rate_wall [1:, c3, i, j]
            Rate_T = Rate_trees[1:, c3, i, j]
            Rate_B = Rate_bikes[1:, c3, i, j]

            color = f'rgba({i * 30}, {j * 20}, {(i + j) * 2}, 1)'
            style = line_styles[j % len(line_styles)]

            legend_group = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_graf[:, c3, i, j]).any() or np.all(Rate_graf[:, c3, i, j]==0)):
                trace_G = go.Scatter(x=x, y=Rate_G, mode='lines', line=dict(color=color, dash=style), name=legend_group, legendgroup=legend_group, showlegend= True)
                fig2.add_trace(trace_G, row=1, col=1)
            if not (np.isnan(Rate_wall[:, c3, i, j]).any() or np.all(Rate_wall[:, c3, i, j]==0)):
                trace_W = go.Scatter(x=x, y=Rate_W, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
                fig2.add_trace(trace_W, row=1, col=2)
            if not (np.isnan(Rate_trees[:, c3, i, j]).any() or np.all(Rate_trees[:, c3, i, j]==0)):
                trace_T = go.Scatter(x=x, y=Rate_T, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
                fig2.add_trace(trace_T, row=2, col=1)
            if not (np.isnan(Rate_bikes[:, c3, i, j]).any() or np.all(Rate_bikes[:, c3, i, j]==0)):
                trace_B = go.Scatter(x=x, y=Rate_B, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
                fig2.add_trace(trace_B, row=2, col=2)

fig2.write_html("./html/oxfordAffineData1234.html")
###########################################################################################################
"""
..#######..##.....##.########..#######..########..########.....########..#######..########..#######.
.##.....##..##...##..##.......##.....##.##.....##.##.....##....##.......##.....##.##....##.##.....##
.##.....##...##.##...##.......##.....##.##.....##.##.....##....##.......##............##...##.....##
.##.....##....###....######...##.....##.########..##.....##....#######..########.....##.....#######.
.##.....##...##.##...##.......##.....##.##...##...##.....##..........##.##.....##...##.....##.....##
.##.....##..##...##..##.......##.....##.##....##..##.....##....##....##.##.....##...##.....##.....##
..#######..##.....##.##........#######..##.....##.########......######...#######....##......#######.
"""
Rate_bark   = np.load(maindir + '/arrays/Rate_bark.npy')
Rate_boat   = np.load(maindir + '/arrays/Rate_boat.npy')
Rate_leuven = np.load(maindir + '/arrays/Rate_leuven.npy')
Rate_ubc    = np.load(maindir + '/arrays/Rate_ubc.npy')

fig4 = make_subplots(rows=2, cols=2, subplot_titles=['Bark(Zoom+Rotation)', 'Boat(Zoom+Rotation)', 'Leuven(Light)', 'UBC(JPEG Compression)'], shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig4.update_layout(margin=dict(l=20, r=20, t=25, b=25))
x = ["Img2", "Img3", "Img4", "Img5", "Img6"]
fig4.update_layout(  xaxis = dict(tickmode = 'array', tickvals = x), xaxis2 = dict(tickmode = 'array', tickvals = x), xaxis3 = dict(tickmode = 'array', tickvals = x), xaxis4 = dict(tickmode = 'array', tickvals = x))
fig4.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig4.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig4.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig4.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate_B  = Rate_bark     [1:, c3, i, j]
            Rate_Bo = Rate_boat     [1:, c3, i, j]
            Rate_L  = Rate_leuven   [1:, c3, i, j]
            Rate_U  = Rate_ubc      [1:, c3, i, j]

            color = f'rgba({i * 30}, {j * 20}, {(i + j) * 2}, 1)'
            style = line_styles[j % len(line_styles)]

            legend_group = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_bark[:, c3, i, j]).any() or np.all(Rate_bark[:, c3, i, j]==0)):
                trace_B = go.Scatter(x=x, y=Rate_B, mode='lines', line=dict(color=color, dash=style), name=legend_group, legendgroup=legend_group, showlegend= True)
                fig4.add_trace(trace_B, row=1, col=1)
            if not (np.isnan(Rate_boat[:, c3, i, j]).any() or np.all(Rate_boat[:, c3, i, j]==0)):
                trace_Bo = go.Scatter(x=x, y=Rate_Bo, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
                fig4.add_trace(trace_Bo, row=1, col=2)
            if not (np.isnan(Rate_leuven[:, c3, i, j]).any() or np.all(Rate_leuven[:, c3, i, j]==0)):
                trace_L = go.Scatter(x=x, y=Rate_L, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
                fig4.add_trace(trace_L, row=2, col=1)
            if not (np.isnan(Rate_ubc[:, c3, i, j]).any() or np.all(Rate_ubc[:, c3, i, j]==0)):
                trace_U = go.Scatter(x=x, y=Rate_U, mode='lines', line=dict(color=color, dash=style), name='', legendgroup=legend_group, showlegend=False)
                fig4.add_trace(trace_U, row=2, col=2)

fig4.write_html("./html/oxfordAffineData5678.html")
###########################################################################################################
"""
..#######..##.....##.########..#######..########..########.....########.####.##.....##.####.##....##..######..
.##.....##..##...##..##.......##.....##.##.....##.##.....##.......##.....##..###...###..##..###...##.##....##.
.##.....##...##.##...##.......##.....##.##.....##.##.....##.......##.....##..####.####..##..####..##.##.......
.##.....##....###....######...##.....##.########..##.....##.......##.....##..##.###.##..##..##.##.##.##...####
.##.....##...##.##...##.......##.....##.##...##...##.....##.......##.....##..##.....##..##..##..####.##....##.
.##.....##..##...##..##.......##.....##.##....##..##.....##.......##.....##..##.....##..##..##...###.##....##.
..#######..##.....##.##........#######..##.....##.########........##....####.##.....##.####.##....##..######..
"""
Exec_time_graf      = np.load(maindir + '/arrays/Exec_time_graf.npy')
Exec_time_wall      = np.load(maindir + '/arrays/Exec_time_wall.npy')
Exec_time_trees     = np.load(maindir + '/arrays/Exec_time_trees.npy')
Exec_time_bikes     = np.load(maindir + '/arrays/Exec_time_bikes.npy')
Exec_time_bark      = np.load(maindir + '/arrays/Exec_time_bark.npy')
Exec_time_boat      = np.load(maindir + '/arrays/Exec_time_boat.npy')
Exec_time_leuven    = np.load(maindir + '/arrays/Exec_time_leuven.npy')
Exec_time_ubc       = np.load(maindir + '/arrays/Exec_time_ubc.npy')

fig3 = make_subplots(rows=2, cols=2, subplot_titles=['Detectors', 'Descriptors', 'Evaluation(matching)'], shared_xaxes=False, shared_yaxes=False, specs=[[{}, {}],[{"colspan": 2}, None]], horizontal_spacing=0.05, vertical_spacing=0.1)
fig3.update_layout(margin=dict(l=20, r=20, t=25, b=25))
# detector time
for i in range(len(DetectorsLegend)):
    mean_detector_time  = np.mean(np.concatenate((Exec_time_graf[:, :, i, :, 0], Exec_time_wall[:, :, i, :, 0], Exec_time_trees[:, :, i, :, 0], Exec_time_bikes[:, :, i, :, 0], Exec_time_bark[:, :, i, :, 0], Exec_time_boat[:, :, i, :, 0], Exec_time_leuven[:, :, i, :, 0], Exec_time_ubc[:, :, i, :, 0]), axis=0))    
    trace_detect = go.Bar(x=[DetectorsLegend[i]], y=[mean_detector_time], name=DetectorsLegend[i], showlegend=True, text=[f'{mean_detector_time:.4f}'], textposition='auto')
    fig3.add_trace(trace_detect, row=1, col=1)
# descriptor time
for j in range(len(DescriptorsLegend)):
    mean_descriptor_time = np.mean(np.concatenate((Exec_time_graf[:, :, :, j, 1], Exec_time_wall[:, :, :, j, 1], Exec_time_trees[:, :, :, j, 1], Exec_time_bikes[:, :, :, j, 1], Exec_time_bark[:, :, :, j, 1], Exec_time_boat[:, :, :, j, 1], Exec_time_leuven[:, :, :, j, 1], Exec_time_ubc[:, :, :, j, 1]), axis=0))
    trace_descr = go.Bar(x=[DescriptorsLegend[j]], y=[mean_descriptor_time], name=DescriptorsLegend[j], showlegend=True, text=[f'{mean_descriptor_time:.4f}'], textposition='auto')
    fig3.add_trace(trace_descr, row=1, col=2)
# matching time
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        mean_matching_time = np.nanmean(np.concatenate((Exec_time_graf[:, :, i, j, 2], Exec_time_wall[:, :, i, j, 2], Exec_time_trees[:, :, i, j, 2], Exec_time_bikes[:, :, i, j, 2], Exec_time_bark[:, :, i, j, 2], Exec_time_boat[:, :, i, j, 2], Exec_time_leuven[:, :, i, j, 2], Exec_time_ubc[:, :, i, j, 2]), axis=0))
        if not (np.isnan(mean_matching_time) or mean_matching_time <= 0):
            trace_match = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[mean_matching_time], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{mean_matching_time:.4f}'], textposition='auto')
            fig3.add_trace(trace_match, row=2, col=1)

fig3.write_html("./html/oxfordAffine_timing.html")
############################################################################################################