import numpy as np
import os
import time
import pandas as pd
import plotly
import plotly.graph_objects as go
from constants import *

def plot(data, filename):
        z, y, x = data
        fig = go.Figure( data=[ go.Scatter3d( x=x, y=y, z=z, mode='markers', marker=dict(size=1.5, color='rgb(200, 0, 0)') ) ] )
        fig.update_layout(scene = dict(
                    xaxis = dict( backgroundcolor="black", gridcolor="white", showbackground=True),
                    yaxis = dict( backgroundcolor="black", gridcolor="white", showbackground=True),
                    zaxis = dict( backgroundcolor="black", gridcolor="white", showbackground=True),
                    )
                  )
        camera = dict( up=dict(x=0, y=1, z=0), eye=dict(x=0, y=0.25, z=2) )
        title = 'Image File: ' + filename
        title = title.replace('_', ' ')
        fig.update_layout(scene_camera=camera, title=title)
        urlPath = output_dir + filename
        plotly.offline.plot(fig, filename=urlPath, auto_open=False)

        print('HTML File Path: ' + urlPath + '\n')

def scale(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
