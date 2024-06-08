import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from define import *

########################
# MARK: - Synthetic Data
########################
Rate_intensity      = np.load('./arrays/Rate_intensity.npy')
Rate_scale          = np.load('./arrays/Rate_scale.npy')
Rate_rot            = np.load('./arrays/Rate_rot.npy')
Exec_time_intensity = np.load('./arrays/Exec_time_intensity.npy')
Exec_time_scale     = np.load('./arrays/Exec_time_scale.npy')
Exec_time_rot       = np.load('./arrays/Exec_time_rot.npy')

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
color_index = 0
for i in range(len(DetectorsLegend)):
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
            if not (np.isnan(Rate_scale[:, c3, i, j, 11]).any()):              
                fig1trace_Scale = go.Scatter(x=scale, y=Rate2_S,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig1, showlegend=False)
                fig1.add_trace(fig1trace_Scale,  row=2, col=1)
            if not (np.isnan(Rate_rot[:, c3, i, j, 11]).any()):
                fig1trace_Rot   = go.Scatter(x=rot,   y=Rate2_R,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig1, showlegend=False)
                fig1.add_trace(fig1trace_Rot,  row=2, col=2)
            color_index += 1
fig1.write_html("./html/SyntheticData.html")

###########################
# MARK: - Inlier Synthetic Timing on 1k features
###########################
fig25 = make_subplots(rows=3, cols=2, subplot_titles=['Average 1k Detect time', 'Average 1k Describe time', 'Average 1k Total time', 'Average 1k Inlier time'], shared_xaxes=False, shared_yaxes=False, specs=[[{}, {}],[{"colspan": 2}, None],[{"colspan": 2}, None]],horizontal_spacing=0.05, vertical_spacing=0.1)
fig25.update_layout(margin=dict(l=20, r=20, t=25, b=25))
fig25.update_yaxes(title_text="seconds", row=1, col=1)
fig25.update_yaxes(title_text="seconds", row=1, col=2)
fig25.update_yaxes(title_text="seconds", row=2, col=1)
fig25.update_yaxes(title_text="seconds", row=3, col=1)
# detector time
for i in range(len(DetectorsLegend)):
    mean_keypoint2_number = (np.nanmean(np.concatenate((Rate_intensity[:, :, i, :, 6], Rate_scale[:, :, i, :, 6], Rate_rot[:, :, i, :, 6]), axis=0)))
    mean_detector_time    = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, i, :, 0], Exec_time_scale[:, :, i, :, 0], Exec_time_rot[:, :, i, :, 0]), axis=0)))
    result = mean_detector_time / mean_keypoint2_number * 1000
    trace_detect_synt = go.Bar(x=[DetectorsLegend[i]], y=[result], name=DetectorsLegend[i], showlegend=True, text=[f'{result:.4f}'], textposition='auto')
    fig25.add_trace(trace_detect_synt, row=1, col=1)
# descriptor time
for j in range(len(DescriptorsLegend)):
    mean_descriptor2_number = (np.nanmean(np.concatenate((Rate_intensity[:, :, :, j, 8], Rate_scale[:, :, :, j, 8], Rate_rot[:, :, :, j, 8]), axis=0)))
    mean_descriptor_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, :, j, 1], Exec_time_scale[:, :, :, j, 1], Exec_time_rot[:, :, :, j, 1]), axis=0)))
    result2 = mean_descriptor_time / mean_descriptor2_number * 1000
    trace_descr_synt = go.Bar(x=[DescriptorsLegend[j]], y=[result2], name=DescriptorsLegend[j], showlegend=True, text=[f'{result2:.4f}'], textposition='auto')
    fig25.add_trace(trace_descr_synt, row=1, col=2)
# matching time
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        average_total_count  = (np.nanmean(np.concatenate((Rate_intensity[:, :, i, j, 10], Rate_scale[:, :, i, j, 10], Rate_rot[:, :, i, j, 10]), axis=0)))
        average_inlier_count = (np.nanmean(np.concatenate((Rate_intensity[:, :, i, j,  9], Rate_scale[:, :, i, j,  9], Rate_rot[:, :, i, j,  9]), axis=0)))
        mean_total_time      = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, i, j, 0], Exec_time_scale[:, :, i, j, 0], Exec_time_rot[:, :, i, j, 0], 
                                                           Exec_time_intensity[:, :, i, j, 1], Exec_time_scale[:, :, i, j, 1], Exec_time_rot[:, :, i, j, 1],
                                                           Exec_time_intensity[:, :, i, j, 2], Exec_time_scale[:, :, i, j, 2], Exec_time_rot[:, :, i, j, 2]), axis=0)))
        result3 = (mean_total_time / average_total_count) * 1000
        result4 = (mean_total_time / average_inlier_count) * 1000 if average_inlier_count > 0 else np.nan
        if not np.isnan(result3):
            trace_match_synt_result3 = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[result3], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{result3:.4f}'], textposition='auto')
            fig25.add_trace(trace_match_synt_result3, row=2, col=1)
        if not np.isnan(result4):
            trace_match_synt_result4 = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[result4], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{result4:.4f}'], textposition='auto')
            fig25.add_trace(trace_match_synt_result4, row=3, col=1)
fig25.write_html("./html/SyntheticData_timing_Average1k.html")
######################
# MARK: - Oxford 1234
######################
Rate_graf        = np.load('./arrays/Rate_graf.npy')
Rate_bikes       = np.load('./arrays/Rate_bikes.npy')
Rate_boat        = np.load('./arrays/Rate_boat.npy')
Rate_leuven      = np.load('./arrays/Rate_leuven.npy')
Exec_time_graf   = np.load('./arrays/Exec_time_graf.npy')
Exec_time_bikes  = np.load('./arrays/Exec_time_bikes.npy')
Exec_time_boat   = np.load('./arrays/Exec_time_boat.npy')
Exec_time_leuven = np.load('./arrays/Exec_time_leuven.npy')

fig3 = make_subplots(rows=2, cols=2, subplot_titles=['Graf(Viewpoint)', 'Bikes(Blur)', 'Boat(Zoom + Rotation)', 'Leuven(Light)'], shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig3.update_layout(margin=dict(l=20, r=20, t=25, b=25))
x = ["Img2", "Img3", "Img4", "Img5", "Img6"]
fig3.update_layout(xaxis = dict(tickmode = 'array', tickvals = x), xaxis2 = dict(tickmode = 'array', tickvals = x), xaxis3 = dict(tickmode = 'array', tickvals = x), xaxis4 = dict(tickmode = 'array', tickvals = x))
fig3.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig3.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig3.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig3.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
color_index = 0
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate_Graf   = Rate_graf  [1:, c3, i, j, 11]
            Rate_Bikes  = Rate_bikes [1:, c3, i, j, 11]
            Rate_Boat   = Rate_boat  [1:, c3, i, j, 11]
            Rate_Leuven = Rate_leuven[1:, c3, i, j, 11]

            color = colors[color_index]
            style = line_styles[j % len(line_styles)]
            legend_groupfig3 = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_graf[1:, c3, i, j, 11]).any()):
                fig3trace_Graf   = go.Scatter(x=x, y=Rate_Graf,    mode='lines', line=dict(color=color, dash=style), name=legend_groupfig3, legendgroup=legend_groupfig3, showlegend=True)
                fig3.add_trace(fig3trace_Graf, row=1, col=1)
            if not (np.isnan(Rate_bikes[1:, c3, i, j, 11]).any()):
                fig3trace_Bikes  = go.Scatter(x=x, y=Rate_Bikes,   mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig3, showlegend=False)
                fig3.add_trace(fig3trace_Bikes, row=2, col=2)
            if not (np.isnan(Rate_boat[1:, c3, i, j, 11]).any()):
                fig3trace_Boat   = go.Scatter(x=x, y=Rate_Boat,   mode='lines', line=dict(color=color, dash=style), name='',                legendgroup=legend_groupfig3, showlegend=False)
                fig3.add_trace(fig3trace_Boat, row=1, col=2)
            if not (np.isnan(Rate_leuven[1:, c3, i, j, 11]).any()):
                fig3trace_Leuven = go.Scatter(x=x, y=Rate_Leuven, mode='lines', line=dict(color=color, dash=style), name='',                legendgroup=legend_groupfig3, showlegend=False)
                fig3.add_trace(fig3trace_Leuven,  row=2, col=1)
            color_index += 1
fig3.write_html("./html/oxfordAffineData1234.html")
######################
# MARK: - Oxford 5678
######################
Rate_wall       = np.load('./arrays/Rate_wall.npy')
Rate_trees      = np.load('./arrays/Rate_trees.npy')
Rate_bark       = np.load('./arrays/Rate_bark.npy')
Rate_ubc        = np.load('./arrays/Rate_ubc.npy')
Exec_time_wall  = np.load('./arrays/Exec_time_wall.npy')
Exec_time_trees = np.load('./arrays/Exec_time_trees.npy')
Exec_time_bark  = np.load('./arrays/Exec_time_bark.npy')
Exec_time_ubc   = np.load('./arrays/Exec_time_ubc.npy')

fig4 = make_subplots(rows=2, cols=2, subplot_titles=['Wall(Viewpoint)', 'Trees(Blur)', 'Bark(Zoom + Rotation)', 'UBC(JPEG)'], shared_xaxes=False, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.1)
fig4.update_layout(margin=dict(l=20, r=20, t=25, b=25))
x = ["Img2", "Img3", "Img4", "Img5", "Img6"]
fig4.update_layout(  xaxis = dict(tickmode = 'array', tickvals = x), xaxis2 = dict(tickmode = 'array', tickvals = x), xaxis3 = dict(tickmode = 'array', tickvals = x), xaxis4 = dict(tickmode = 'array', tickvals = x))
fig4.update_yaxes(title_text="Correctly matched point rates %", row=1, col=1)
fig4.update_yaxes(title_text="Correctly matched point rates %", row=1, col=2)
fig4.update_yaxes(title_text="Correctly matched point rates %", row=2, col=1)
fig4.update_yaxes(title_text="Correctly matched point rates %", row=2, col=2)
color_index = 0
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        for c3 in range(len(Norm)):
            Rate_Wall  = Rate_wall [1:, c3, i, j, 11]
            Rate_Trees = Rate_trees[1:, c3, i, j, 11]
            Rate_Bark  = Rate_bark [1:, c3, i, j, 11]
            Rate_Ubc   = Rate_ubc  [1:, c3, i, j, 11]

            color = colors[color_index]
            style = line_styles[j % len(line_styles)]
            legend_groupfig4 = f'{DetectorsLegend[i]}-{DescriptorsLegend[j]}-{Norm[c3]}'
            if not (np.isnan(Rate_bark[1:, c3, i, j, 11]).any()):
                fig4trace_Bark  = go.Scatter(x=x, y=Rate_Bark, mode='lines', line=dict(color=color, dash=style), name=legend_groupfig4, legendgroup=legend_groupfig4, showlegend=True)
                fig4.add_trace(fig4trace_Bark,  row=1, col=1)
            if not (np.isnan(Rate_ubc[1:, c3, i, j, 11]).any()):
                fig4trace_Ubc   = go.Scatter(x=x, y=Rate_Ubc,  mode='lines', line=dict(color=color, dash=style), name='',               legendgroup=legend_groupfig4, showlegend=False)
                fig4.add_trace(fig4trace_Ubc,  row=2, col=2)
            if not (np.isnan(Rate_wall[1:, c3, i, j, 11]).any()):
                fig4trace_Wall  = go.Scatter(x=x, y=Rate_Wall,  mode='lines', line=dict(color=color, dash=style), name='',              legendgroup=legend_groupfig4, showlegend=False)
                fig4.add_trace(fig4trace_Wall, row=1, col=2)
            if not (np.isnan(Rate_trees[1:, c3, i, j, 11]).any()):
                fig4trace_Trees = go.Scatter(x=x, y=Rate_Trees, mode='lines', line=dict(color=color, dash=style), name='',              legendgroup=legend_groupfig4, showlegend=False)
                fig4.add_trace(fig4trace_Trees, row=2, col=1)
            color_index += 1
fig4.write_html("./html/oxfordAffineData5678.html")
###########################
# MARK: - Inlier Oxford Timing on 100k features
###########################
fig55 = make_subplots(rows=3, cols=2, subplot_titles=['Average 1k Detect time', 'Average 1k Describe time', 'Average 1k Total time (Detect + Descript + Match + RANSAC)', 'Average 1k Inlier time (Detect + Descript + Match + RANSAC)'], shared_xaxes=False, shared_yaxes=False, specs=[[{}, {}],[{"colspan": 2}, None],[{"colspan": 2}, None]],horizontal_spacing=0.05, vertical_spacing=0.1)
fig55.update_layout(margin=dict(l=20, r=20, t=25, b=25))
fig55.update_yaxes(title_text="seconds", row=1, col=1)
fig55.update_yaxes(title_text="seconds", row=1, col=2)
fig55.update_yaxes(title_text="seconds", row=2, col=1)
fig55.update_yaxes(title_text="seconds", row=3, col=1)
# detector time
for i in range(len(DetectorsLegend)):
    mean_keypoint2_number = np.nanmean(np.concatenate((Rate_graf[:, :, i, :, 6], Rate_wall[:, :, i, :, 6], Rate_trees[:, :, i, :, 6], Rate_bikes[:, :, i, :, 6], Rate_bark[:, :, i, :, 6], Rate_boat[:, :, i, :, 6], Rate_leuven[:, :, i, :, 6], Rate_ubc[:, :, i, :, 6]), axis=0))
    mean_detector_time    = np.nanmean(np.concatenate((Exec_time_graf[:, :, i, :, 0], Exec_time_wall[:, :, i, :, 0], Exec_time_trees[:, :, i, :, 0], Exec_time_bikes[:, :, i, :, 0], Exec_time_bark[:, :, i, :, 0], Exec_time_boat[:, :, i, :, 0], Exec_time_leuven[:, :, i, :, 0], Exec_time_ubc[:, :, i, :, 0]), axis=0))    
    result = mean_detector_time / mean_keypoint2_number * 1000
    trace_detect_oxford = go.Bar(x=[DetectorsLegend[i]], y=[result], name=DetectorsLegend[i], showlegend=True, text=[f'{result:.4f}'], textposition='auto')
    fig55.add_trace(trace_detect_oxford, row=1, col=1)
# descriptor time
for j in range(len(DescriptorsLegend)):
    mean_descriptor2_number = np.nanmean(np.concatenate((Rate_graf[:, :, :, j, 8], Rate_wall[:, :, :, j, 8], Rate_trees[:, :, :, j, 8], Rate_bikes[:, :, :, j, 8], Rate_bark[:, :, :, j, 8], Rate_boat[:, :, :, j, 8], Rate_leuven[:, :, :, j, 8], Rate_ubc[:, :, :, j, 8]), axis=0))
    mean_descriptor_time    = np.nanmean(np.concatenate((Exec_time_graf[:, :, :, j, 1], Exec_time_wall[:, :, :, j, 1], Exec_time_trees[:, :, :, j, 1], Exec_time_bikes[:, :, :, j, 1], Exec_time_bark[:, :, :, j, 1], Exec_time_boat[:, :, :, j, 1], Exec_time_leuven[:, :, :, j, 1], Exec_time_ubc[:, :, :, j, 1]), axis=0))
    result2 = mean_descriptor_time / mean_descriptor2_number * 1000
    trace_descr_oxford = go.Bar(x=[DescriptorsLegend[j]], y=[result2], name=DescriptorsLegend[j], showlegend=True, text=[f'{result2:.4f}'], textposition='auto')
    fig55.add_trace(trace_descr_oxford, row=1, col=2)
# matching time
for i in range(len(DetectorsLegend)):
    for j in range(len(DescriptorsLegend)):
        average_total_count  = np.nanmean(np.concatenate((Rate_graf[:, :, i, j, 10], Rate_wall[:, :, i, j, 10], Rate_trees[:, :, i, j, 10], Rate_bikes[:, :, i, j, 10], Rate_bark[:, :, i, j, 10], Rate_boat[:, :, i, j, 10], Rate_leuven[:, :, i, j, 10], Rate_ubc[:, :, i, j, 10]), axis=0))
        average_inlier_count = np.nanmean(np.concatenate((Rate_intensity[:, :, i, j,  9], Rate_scale[:, :, i, j,  9], Rate_rot[:, :, i, j,  9]), axis=0))
        mean_total_time      = np.nanmean(np.concatenate((Exec_time_graf[:, :, i, j, 0], Exec_time_wall[:, :, i, j, 0], Exec_time_trees[:, :, i, j, 0], Exec_time_bikes[:, :, i, j, 0], Exec_time_bark[:, :, i, j, 0], Exec_time_boat[:, :, i, j, 0], Exec_time_leuven[:, :, i, j, 0], Exec_time_ubc[:, :, i, j, 0], 
                                                           Exec_time_graf[:, :, i, j, 1], Exec_time_wall[:, :, i, j, 1], Exec_time_trees[:, :, i, j, 1], Exec_time_bikes[:, :, i, j, 1], Exec_time_bark[:, :, i, j, 1], Exec_time_boat[:, :, i, j, 1], Exec_time_leuven[:, :, i, j, 1], Exec_time_ubc[:, :, i, j, 1],
                                                           Exec_time_graf[:, :, i, j, 2], Exec_time_wall[:, :, i, j, 2], Exec_time_trees[:, :, i, j, 2], Exec_time_bikes[:, :, i, j, 2], Exec_time_bark[:, :, i, j, 2], Exec_time_boat[:, :, i, j, 2], Exec_time_leuven[:, :, i, j, 2], Exec_time_ubc[:, :, i, j, 2]), axis=0))
        result3 = (mean_total_time / average_total_count) * 1000 if average_total_count > 0 else np.nan
        result4 = (mean_total_time / average_inlier_count) * 1000 if average_inlier_count > 0 else np.nan
        if not np.isnan(result3):
            trace_match_oxford_result3 = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[result3], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{result3:.4f}'], textposition='auto')
            fig55.add_trace(trace_match_oxford_result3, row=2, col=1)
        if not np.isnan(result4):
            trace_match_oxford_result4 = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[result4], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{result4:.4f}'], textposition='auto')
            fig55.add_trace(trace_match_oxford_result4, row=3, col=1)
fig55.write_html("./html/oxfordAffine_timing_Average1k.html")











# ###########################
# # MARK: - Synthetic Timing
# ###########################
# fig2 = make_subplots(rows=2, cols=2, subplot_titles=['Detectors', 'Descriptors', 'Matching+XSAC'], shared_xaxes=False, shared_yaxes=False, specs=[[{}, {}],[{"colspan": 2}, None]],horizontal_spacing=0.05, vertical_spacing=0.1)
# fig2.update_layout(margin=dict(l=20, r=20, t=25, b=25))
# fig2.update_yaxes(title_text="seconds", row=1, col=1)
# fig2.update_yaxes(title_text="seconds", row=1, col=2)
# fig2.update_yaxes(title_text="seconds", row=2, col=1)
# # detector time
# for i in range(len(DetectorsLegend)):
#     mean_detector_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, i, :, 0], Exec_time_scale[:, :, i, :, 0], Exec_time_rot[:, :, i, :, 0]), axis=0)))
#     trace_detect_synt = go.Bar(x=[DetectorsLegend[i]], y=[mean_detector_time], name=DetectorsLegend[i], showlegend=True, text=[f'{mean_detector_time:.4f}'], textposition='auto')
#     fig2.add_trace(trace_detect_synt, row=1, col=1)
# # descriptor time
# for j in range(len(DescriptorsLegend)):
#     mean_descriptor_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, :, j, 1], Exec_time_scale[:, :, :, j, 1], Exec_time_rot[:, :, :, j, 1]), axis=0)))
#     trace_descr_synt = go.Bar(x=[DescriptorsLegend[j]], y=[mean_descriptor_time], name=DescriptorsLegend[j], showlegend=True, text=[f'{mean_descriptor_time:.4f}'], textposition='auto')
#     fig2.add_trace(trace_descr_synt, row=1, col=2)
# # matching time
# for i in range(len(DetectorsLegend)):
#     for j in range(len(DescriptorsLegend)):
#         mean_matching_time = (np.nanmean(np.concatenate((Exec_time_intensity[:, :, i, j, 2], Exec_time_scale[:, :, i, j, 2], Exec_time_rot[:, :, i, j, 2]), axis=0)))
#         if not (np.isnan(mean_matching_time) or mean_matching_time <= 0):
#             trace_match_synt = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[mean_matching_time], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{mean_matching_time:.4f}'], textposition='auto')
#             fig2.add_trace(trace_match_synt, row=2, col=1)
# fig2.write_html("./html/SyntheticData_timing.html")
# ########################
# # MARK: - Oxford Timing
# ########################
# fig5 = make_subplots(rows=2, cols=2, subplot_titles=['Detectors', 'Descriptors', 'Matching+XSAC'], shared_xaxes=False, shared_yaxes=False, specs=[[{}, {}],[{"colspan": 2}, None]], horizontal_spacing=0.05, vertical_spacing=0.1)
# fig5.update_layout(margin=dict(l=20, r=20, t=25, b=25))
# fig5.update_yaxes(title_text="seconds", row=1, col=1)
# fig5.update_yaxes(title_text="seconds", row=1, col=2)
# fig5.update_yaxes(title_text="seconds", row=2, col=1)
# fig5.update_yaxes(title_text="seconds", row=3, col=1)
# # detector time
# for i in range(len(DetectorsLegend)):
#     mean_detector_time = np.nanmean(np.concatenate((Exec_time_graf[:, :, i, :, 0], Exec_time_wall[:, :, i, :, 0], Exec_time_trees[:, :, i, :, 0], Exec_time_bikes[:, :, i, :, 0], Exec_time_bark[:, :, i, :, 0], Exec_time_boat[:, :, i, :, 0], Exec_time_leuven[:, :, i, :, 0], Exec_time_ubc[:, :, i, :, 0]), axis=0))    
#     trace_detect_oxf   = go.Bar(x=[DetectorsLegend[i]], y=[mean_detector_time], name=DetectorsLegend[i], showlegend=True, text=[f'{mean_detector_time:.4f}'], textposition='auto')
#     fig5.add_trace(trace_detect_oxf, row=1, col=1)
# # descriptor time
# for j in range(len(DescriptorsLegend)):
#     mean_descriptor_time = np.nanmean(np.concatenate((Exec_time_graf[:, :, :, j, 1], Exec_time_wall[:, :, :, j, 1], Exec_time_trees[:, :, :, j, 1], Exec_time_bikes[:, :, :, j, 1], Exec_time_bark[:, :, :, j, 1], Exec_time_boat[:, :, :, j, 1], Exec_time_leuven[:, :, :, j, 1], Exec_time_ubc[:, :, :, j, 1]), axis=0))
#     trace_descr_oxf      = go.Bar(x=[DescriptorsLegend[j]], y=[mean_descriptor_time], name=DescriptorsLegend[j], showlegend=True, text=[f'{mean_descriptor_time:.4f}'], textposition='auto')
#     fig5.add_trace(trace_descr_oxf, row=1, col=2)
# # matching time
# for i in range(len(DetectorsLegend)):
#     for j in range(len(DescriptorsLegend)):
#         mean_matching_time  = np.nanmean(np.concatenate((Exec_time_graf[:, :, i, j, 2], Exec_time_wall[:, :, i, j, 2], Exec_time_trees[:, :, i, j, 2], Exec_time_bikes[:, :, i, j, 2], Exec_time_bark[:, :, i, j, 2], Exec_time_boat[:, :, i, j, 2], Exec_time_leuven[:, :, i, j, 2], Exec_time_ubc[:, :, i, j, 2]), axis=0))
#         if not (np.isnan(mean_matching_time) or mean_matching_time <= 0):
#             trace_match_oxf = go.Bar(x=[DetectorsLegend[i] + '-' + DescriptorsLegend[j]], y=[mean_matching_time], name=DetectorsLegend[i] + '-' + DescriptorsLegend[j], showlegend=True, text=[f'{mean_matching_time:.4f}'], textposition='auto')
#             fig5.add_trace(trace_match_oxf, row=2, col=1)
# fig5.write_html("./html/oxfordAffine_timing.html")