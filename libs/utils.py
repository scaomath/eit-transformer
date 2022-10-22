import gc
import os
import sys
import yaml
import argparse
import math
import pickle
import copy
import random as rd
from collections import defaultdict
from contextlib import contextmanager
from datetime import date
from time import time

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, tri
import numpy as np
import pandas as pd
import psutil
import platform
import subprocess
import re
import torch
import torch.nn as nn

import h5py
from scipy.io import loadmat

try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as e:
    print('Plotly not installed. Plotly is used to plot meshes')


def get_size(bytes, suffix='B'):
    ''' 
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MiB'
        1253656678 => '1.17GiB'
    '''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.2f} {unit}{suffix}"
        bytes /= 1024.0
    return f"{bytes:3.2f} 'Yi'{suffix}"


def get_file_size(filename):
    file_size = os.stat(filename)
    return get_size(file_size.st_size)


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.decode("utf-8").split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)


def get_system():
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Device name       :", get_processor_name())
    print("Physical cores    :", psutil.cpu_count(logical=False))
    print("Total cores       :", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency    : {cpufreq.max:.2f} Mhz")
    print(f"Min Frequency    : {cpufreq.min:.2f} Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f} Mhz")

    print("="*40, "Memory Info", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total     : {get_size(svmem.total)}")
    print(f"Available : {get_size(svmem.available)}")
    print(f"Used      : {get_size(svmem.used)}")

    print("="*40, "Software Info", "="*40)
    print('Python     : ' + sys.version.split('\n')[0])
    print('Numpy      : ' + np.__version__)
    print('Pandas     : ' + pd.__version__)
    print('PyTorch    : ' + torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print("="*40, "GPU Info", "="*40)
        print(f'Device     : {torch.cuda.get_device_name(0)}')
        print(f"{'Mem total': <15}: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
        print(
            f"{'Mem allocated': <15}: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(
            f"{'Mem cached': <15}: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

    print("="*30, "system info print done", "="*30)


def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    '''
    if cudnn:
        message += f'''
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        '''

    if torch.cuda.is_available():
        message += f'''
        torch.cuda.manual_seed_all({s})
        '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("="*50)
        print(message)
        print("="*50)


@contextmanager
def simple_timer(title):
    t0 = time()
    yield
    print("{} - done in {:.1f} seconds.\n".format(title, time() - t0))


class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer(label: str, compact=False) -> None:
    '''
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    print 
    1. the time the code block takes to run
    2. the memory usage.
    '''
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    start = time()  # Setup - __enter__
    if not compact:
        print(color(f"{label}: start at {start:.2f};", color=Colors.blue))
        print(
            color(f"LOCAL RAM USAGE AT START: {m0:.2f} GB", color=Colors.green))
        try:
            yield  # yield to body of `with` statement
        finally:  # Teardown - __exit__
            m1 = p.memory_info()[0] / 2. ** 30
            delta = m1 - m0
            sign = '+' if delta >= 0 else '-'
            delta = math.fabs(delta)
            end = time()
            print(color(
                f"{label}: done at {end:.2f} ({end - start:.6f} secs elapsed);", color=Colors.blue))
            print(color(
                f"LOCAL RAM USAGE AT END: {m1:.2f}GB ({sign}{delta:.2f}GB)", color=Colors.green))
            print('\n')
    else:
        yield
        print(
            color(f"{label} - done in {time() - start:.6f} seconds. \n", color=Colors.blue))


def get_memory(num_var=10):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:num_var]:
        print(color(f"{name:>30}:", color=Colors.green),
              color(f"{get_size(size):>8}", color=Colors.magenta))


def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for _file in files:
            if name in _file:
                result.append(os.path.join(root, _file))
    return sorted(result)


def print_file_size(files):
    for file in files:
        size = get_file_size(file)
        filename = file.split('/')[-1]
        filesize = get_file_size(file)
        print(color(f"{filename:>30}:", color=Colors.green),
              color(f"{filesize:>8}", color=Colors.magenta))


@contextmanager
def trace(title: str):
    t0 = time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB ({sign}{delta:.3f}GB): {time() - t0:.2f}sec] {title} ", file=sys.stderr)


def get_cmap(n, cmap='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmap, n)


def get_date():
    today = date.today()
    return today.strftime("%b-%d-%Y")


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
        # num_params += p.numel() * (1 + p.is_complex())
    return num_params

def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        u = pickle.load(f)
    return u


def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value


class DotDict(dict):
    """
    https://stackoverflow.com/a/23689767/622119
    https://stackoverflow.com/a/36968114/622119
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

def load_yaml(filename, key=None):
    with open(filename) as f:
        dictionary = yaml.full_load(f)
    if key is None:
        return DotDict(dictionary)
    else:
        return DotDict(dictionary[key])

def showmesh(node, elem, **kwargs):
    triangulation = tri.Triangulation(node[:, 0], node[:, 1], elem)
    markersize = 3000/len(node)
    if kwargs.items():
        h = plt.triplot(triangulation, 'b-h', **kwargs)
    else:
        h = plt.triplot(triangulation, 'b-h', linewidth=0.5,
                        alpha=0.5, markersize=markersize)
    return h


def showsolution(node, elem, u, **kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    on triangulations
    '''
    markersize = 3000/len(node)

    if u.ndim == 1:  # (N, )
        uplot = ff.create_trisurf(x=node[:, 0], y=node[:, 1], z=u,
                                  simplices=elem,
                                  colormap="Viridis",  # similar to matlab's default colormap
                                  showbackground=True,
                                  show_colorbar=False,
                                  aspectratio=dict(x=1, y=1, z=1),
                                  )
        fig = go.Figure(data=uplot)

    elif u.ndim == 2 and u.shape[1] == 2:  # (N, 2)
        if u.shape[0] == elem.shape[0]:
            u /= (np.abs(u)).max()
            node = node[elem].mean(axis=1)

        uplot = ff.create_quiver(x=node[:, 0], y=node[:, 1],
                                 u=u[:, 0], v=u[:, 1],
                                 scale=.2,
                                 arrow_scale=.5,
                                 name='gradient of u',
                                 line_width=1,
                                 )

        fig = go.Figure(data=uplot)

    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    fig.show()
    return fig


def showsurf(x, y, z, **kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    on a meshgrid
    x, y, z: (M, N) matrix
    '''

    uplot = go.Surface(x=x, y=y, z=z,
                       colorscale='Viridis',
                       showscale=False),

    fig = go.Figure(data=uplot)

    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    fig.show()


def showcontour(z, **kwargs):
    '''
    show 2D solution z of its contour
    '''
    uplot = go.Contour(z=z,
                       colorscale='RdYlBu',
                       #    colorscale='balance',
                       line_smoothing=0.85,
                       line_width=0.1,
                       #    contours_coloring ='heatmap',
                       #    showlabels = True,
                       #    contours=dict(
                       #        coloring='heatmap',
                       #        showlabels=True,  # show labels on contours
                       #        # labelfont = dict( # label font properties
                       #        #     size = 12,
                       #        #     color = 'white',
                       #    )
                       )
    fig = go.Figure(data=uplot,
                    layout={'xaxis': {'title': 'x-label',
                                      'visible': False,
                                      'showticklabels': False},
                            'yaxis': {'title': 'y-label',
                                      'visible': False,
                                      'showticklabels': False}},)
    fig.update_traces(showscale=False)
    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    fig.show()
    return fig


if __name__ == "__main__":
    get_system()
    get_memory()
