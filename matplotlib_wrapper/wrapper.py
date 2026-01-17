'''
Docstring for matplotlib_wrapper.wrapper

This tiny project is a wrapper for some mostly used features in matplotlib.
Literally a wrapper. 
If you don't use numpy or torch, simply commend the corresponding parts out.
'''
import numpy
import torch
type data_list = list|numpy.ndarray|torch.Tensor
# You have to manually do this, bc type definition must be able to resolve in compile time.
# If my understanding is wrong, post issue, bc I don't know the trick.
# But don't post pr. You know the reason.
# I know the multiple file trick, but I don't want to implement it here. At least not now.

import sys

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"






from matplotlib import pyplot as plt
from matplotlib import artist
from matplotlib import lines 
from matplotlib import axes
from matplotlib import figure
from matplotlib import legend


from typing import Literal
if "marker type defination" and "bc I want to fold it":
    type marker_type_raw = Literal[
            '.','point',',','pixel','o','circle',
            'v','triangle_down','^','triangle_up','<','triangle_left','>','triangle_right',
            '1','tri_down','2','tri_up','3','tri_left','4','tri_right','8','octagon',
            's','square','p','pentagon','P','plus (filled)',
            '*','star','h','hexagon1','H','hexagon2',
            '+','plus','x','x','X','x (filled)',
            'D','diamond','d','thin_diamond',
            '|','vline','_','hline',
            None]
    type marker_type = Literal[
            '.',',','o','v','^','<','>',
            '1','2','3','4','8','s','p','P',
            '*','h','H','+','x','X','D','d','|','_',
            None]
    pass
def correct_marker(input:marker_type_raw)->marker_type:
    if input in [
            '.',',','o','v','^','<','>',
            '1','2','3','4','8','s','p','P',
            '*','h','H','+','x','X','D','d','|','_']:
        return input
    match input:
        case 'point':              
            return '.'
        case 'pixel':              
            return ','
        case 'circle':             
            return 'o'
        case 'triangle_down':      
            return 'v'
        case 'triangle_up':        
            return '^'
        case 'triangle_left':      
            return '<'
        case 'triangle_right':     
            return '>'
        case 'tri_down':           
            return '1'
        case 'tri_up':             
            return '2'
        case 'tri_left':           
            return '3'
        case 'tri_right':          
            return '4'
        case 'octagon':            
            return '8'
        case 'square':             
            return 's'
        case 'pentagon':           
            return 'p'
        case 'plus (filled)':      
            return 'P'
        case 'star':               
            return '*'
        case 'hexagon1':           
            return 'h'
        case 'hexagon2':           
            return 'H'
        case 'plus':               
            return '+'
        case 'x':                  
            return 'x'
        case 'x (filled)':         
            return 'X'
        case 'diamond':            
            return 'D'
        case 'thin_diamond':       
            return 'd'
        case 'vline':              
            return '|'
        case 'hline':              
            return '_'
        case None:
            return ''
        case _:
            assert False, "bad input."
            pass
        #pass#/ match
    pass#/ function.

if "line type defination" and "bc I want to fold it":
    type line_type_raw = Literal['-','solid','--','dashed','-.','dash-dot',':','dotted',
            None]
    type line_type = Literal['-','--','-.',':',
            None]
    pass
def correct_line(input:line_type_raw)->line_type:
    if input in ['-','--','-.',':']:
        return input
    match input:
        case 'solid':
            return '-'         
        case 'dashed':
            return '--'        
        case 'dash-dot':
            return '-.'        
        case 'dotted':
            return ':' 
        case None:
            return ''
        case _:
            assert False, "bad input."
            pass
        #pass#/ match
    pass#/ function.

if "color type defination" and "bc I want to fold it":
    type color_type_raw = Literal['b','blue','g','green','r','red',
                        'c','cyan','m','magenta','y','yellow','k','black',
                        'w','white',
            None]
    type color_type = Literal['b','g','r',
                            'c','m','y','k',
                            'w',
            None]
    pass
def correct_color(input:color_type_raw)->color_type:
    if input in ['b','g','r',
                'c','m','y','k',
                'w']:
        return input
    match input:
        case 'blue':    
            return 'b'
        case 'green':   
            return 'g'
        case 'red':     
            return 'r'
        case 'cyan':    
            return 'c'
        case 'magenta': 
            return 'm'
        case 'yellow':  
            return 'y'
        case 'black':   
            return 'k'
        case 'white':   
            return 'w'
        case None:
            return ''
        case _:
            assert False, "bad input."
            pass
        #pass#/ match
    pass#/ function.

def torch_Tensor_to_list(tensor:torch.Tensor)->list:
    return tensor.tolist()

def get_current_figure()->figure.Figure:
    return plt.gcf()

def get_current_axes()->axes.Axes:
    return plt.gca()

def any_1d_list_to_pylist(input:data_list|None)->list:
    if "numpy" in sys.modules:
        if isinstance(input, numpy.ndarray):
            return input.tolist()
        pass
    if "torch" in sys.modules:
        if isinstance(input, torch.Tensor):
            return input.tolist()
        pass
    assert isinstance(input, list) or input is None, f"is this a numpy.ndarray or torch.Tensor? {None\
                            }Umcomment the corresponding import instruct on the top."
    return input

def LineChart_on_axes(axes:axes.Axes = get_current_axes(), data_x:data_list = [], data_y:data_list = [], label = "", \
                marker:marker_type_raw = None, line:line_type_raw = None, color:color_type_raw = None, \
                        linewidth=2, markersize=12 ,\
                        scalex = True, scaley = True)->list[lines.Line2D]:
    #all the types:
    data_x_inner = any_1d_list_to_pylist(data_x)
    data_y_inner = any_1d_list_to_pylist(data_y)
    assert data_y_inner.__len__()>0,"At least some data to plot, right?"
    
    marker_inner:marker_type|str = correct_marker(marker)
    line_inner:line_type|str = correct_line(line)
    color_inner:color_type|str = correct_color(color)
    
    fmt = f'{marker_inner}{line_inner}{color_inner}'
    
    return axes.plot(data_x_inner, data_y_inner, fmt, label = label, 
                    linewidth=linewidth, markersize=markersize, 
                    scalex = scalex, scaley = scaley)

def set_legend_according_to_artists()->legend.Legend:
    '''collect all the legend info and add a real legend object into axes.'''
    return plt.legend()
    
def set_legend(handles:list[artist.Artist]|None = None, 
                                names:list[str]|None = None)->legend.Legend:
    return plt.legend(handles, names)
    
    
if "test" and __DEBUG_ME__() and True:
    data_1_x = torch.tensor([1,2])
    data_1_y = torch.tensor([1,3])
    data_2_x = torch.tensor([5,6])
    data_2_y = torch.tensor([5,7])

    lines_test:list[artist.Artist] = []
    lines_test.extend(LineChart_on_axes(get_current_axes(), data_1_x, data_1_y,
                            "data 1111", marker="+", line="--"))
    LineChart_on_axes(get_current_axes(), data_2_x, data_2_y, 
                    "data 2222", marker="2", line='-.', markersize=15)
    set_legend(lines_test,["11111111"])
    
    #set_legend_according_to_artists()
    
    plt.show()
    pass










1w 继续。


https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig, ax = plt.subplots()
im = ax.imshow(harvest)






import matplotlib as mpl


vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(farmers)), labels=farmers,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(vegetables)), labels=vegetables)

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()













#plt.plot(x,y)

plt.Axes.plot
plt.Line2D
plt.gca()
plt.gca().plot()#list[Line2D]


plt.gcf()# Figure
plt.gcf().get_figure()#Return the .Figure or .SubFigure instance the (Sub)Figure belongs to.
plt.gcf().get_axes()# list[Axes]
plt.gcf().get_children()# list[Artist]

plt.gca()# Axes
plt.gca().get_figure()#Return the .Figure or .SubFigure instance the artist belongs to.
plt.gca().get_children()# list[Artist]








#     matplotlib.pyplot (State Machine Interface)
#         The top-level interface (plt) acts like a state machine, keeping track of the current figure and axes.
#         It creates and stores references to figures in memory until they are explicitly closed.

#     Figure
#         Represents the entire drawing canvas.
#         A Figure object can contain multiple Axes objects.
#         Memory note: Large figures with many subplots or images can consume significant memory.

#     Axes
#         The actual plotting area (what you think of as a “plot”).
#         Each Axes contains:
#             Axis objects (X-axis, Y-axis)
#             Artists (lines, text, patches, images, etc.)

#     Axis
#         Manages ticks, tick labels, and axis labels.
#         Stores references to tick objects and text labels.

#     Artist (Base Class for All Visible Elements)
#         Everything drawn in Matplotlib is an Artist.
#         Examples:
#             Line2D (lines)
#             Text (labels, titles)
#             Patch (shapes)
#             Image (raster data)
#         Artists are stored in lists inside Axes and Figure.





#     fig, ax = plt.subplots()
#     ax.plot([1, 2, 3])
#     plt.close(fig)  # Frees memory for this figure

#     Repeated plotting without closing can cause memory leaks.
#     Use:
#         plt.clf() → Clear current figure
#         plt.cla() → Clear current axes
#         plt.close() → Close figure and release memory

# Hierarchy Diagram

# Copy code
# pyplot (state machine)
#    └── Figure(s)
#          └── Axes
#                ├── Axis (X, Y)
#                └── Artists (Line2D, Text, Patch, Image, etc.)

