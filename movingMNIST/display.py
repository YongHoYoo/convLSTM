import sys
import torch 
import pickle
import argparse
from pathlib import Path

import plotly 
import plotly.graph_objs as go
from plotly import tools 

if __name__ == '__main__': 

#    parser = argparse.ArgumentParser(description='Argument Parser') 
#    parser.add_argument('--save_folder', type=str, default='00') 
#
#    args = parser.parse_args() 

#    save_folder = Path(args.save_folder) 

    traces = [] 
    save_folder = ['00', '01', '10', '11'] 

    for folder in save_folder: 

        sub_folder = Path(folder)
        valid_filename = sub_folder.joinpath('valid.pkl') 

        valid = pickle.load(open(str(sub_folder.joinpath('valid.pkl')), 'rb')) 
    
        trace = go.Scatter(
            x = list(range(len(valid))),        
            y = valid, 
            mode = 'lines+markers', 
            name = folder, 
        ) 

        traces.append(trace) 

    plotly.offline.plot({
        'data': traces, 
        'layout': go.Layout(title='valid', plot_bgcolor='rgb(239,239,239')}, 
        filename='valid_error.html') 


       
