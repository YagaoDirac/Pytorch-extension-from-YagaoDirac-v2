'''
Docstring for matplotlib_wrapper.wrapper

This tiny project is a wrapper for some mostly used features in matplotlib.
Literally a wrapper. 
If you don't use numpy or torch, simply commend the corresponding parts out.
'''
import numpy
import torch
type data_list = list|numpy.ndarray|torch.Tensor
type data_list_2d = list[list]|numpy.ndarray|torch.Tensor
# You have to manually do this, bc type definition must be able to resolve in compile time.
# If my understanding is wrong, post issue, bc I don't know the trick.
# But don't post pr. You know the reason.
# I know the multiple file trick, but I don't want to implement it here. At least not now.

import sys

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import artist
from matplotlib import lines 
from matplotlib import axes
from matplotlib import figure
from matplotlib import legend
from matplotlib import axis
from matplotlib import text
from matplotlib import image
from matplotlib import colors
from matplotlib import contour
from matplotlib import markers

#from matplotlib.colors import Colormap

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
            '|','vline','_','hline']
    type marker_type = Literal[#do I need this???
            '.',',','o','v','^','<','>',
            '1','2','3','4','8','s','p','P',
            '*','h','H','+','x','X','D','d','|','_']
    pass
def correct_marker(input:marker_type_raw|None)->str:#marker_type:
    if input is None:
        return ''
    if input in [
            '.',',','o','v','^','<','>',
            '1','2','3','4','8','s','p','P',
            '*','h','H','+','x','X','D','d','|','_']:
        #assert isinstance(input, str)
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
matplotlib_default_marker_style_list = markers.MarkerStyle.markers

if "line type defination" and "bc I want to fold it":
    type line_type_raw = Literal['-','solid','--','dashed','-.','dash-dot',':','dotted']
    type line_type = Literal['-','--','-.',':']#do I need this???
    pass
def correct_line(input:line_type_raw|None)->str:#line_type:
    if input is None:
        return ''
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
matplotlib_default_line_style_list = lines.Line2D.lineStyles

if "color type defination" and "bc I want to fold it":
    type color_type_raw = Literal['b','blue','g','green','r','red',
                        'c','cyan','m','magenta','y','yellow','k','black',
                        'w','white']
    type color_type = Literal['b','g','r',#do I need this???
                            'c','m','y','k',
                            'w']
    pass
matplotlib_default_color_list = colors.cnames

def correct_color(input:color_type_raw|None|str)->str:#color_type:
    if input is None:
        return ''
    
    # if input.__len__() == 7:
    #     assert input[0] == '#'
    #     for ii in range(1,7):
    #         assert input[ii] in "012345789abcdefABCDEF"
    #         pass
    #     return input
    
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

def any_1d_list_to_pylist(input:data_list)->list:
    if "numpy" in sys.modules:
        if isinstance(input, numpy.ndarray):
            return input.tolist()
        pass
    if "torch" in sys.modules:
        if isinstance(input, torch.Tensor):
            return input.tolist()
        pass
    assert isinstance(input, list), f"is this a numpy.ndarray or torch.Tensor? {None\
                            }Umcomment the corresponding import instruct on the top."
    return input



from contextlib import ExitStack
def wiggle_and_fun(scale: float = 1,length: float = 100,randomness: float = 2)-> ExitStack[bool | None]:
    return plt.xkcd()

def set_style(index:int)->tuple[str,list[str]]:
    '''return new_style, all_available_style'''
    new_style = plt.style.available[index]
    plt.style.use(new_style)
    return new_style, plt.style.available



def get_current_figure()->figure.Figure:
    return plt.gcf()

def get_current_axes()->axes.Axes:
    return plt.gca()



def LineChart_on_axes(data_x:data_list|None = None, data_y:data_list = [], label = "", \
                marker:marker_type_raw|None = None, line:line_type_raw|None = None, \
                color:color_type_raw|None = None, color_in_hex_format:str|None = "#000000", \
                linewidth=2, markersize=12 ,\
                scalex = True, scaley = True,the_axes:axes.Axes = get_current_axes(), )->list[lines.Line2D]:
    '''The color can be in either format. To use the matplotlib preset color, only set color.
    To use the hex color format, set color to none, and set color_in_hex_format to any color.
    '''
    #marker line color:
    marker_inner:str = correct_marker(marker)
    line_inner:str = correct_line(line)
    if color is None:
        color_inner:str = ""
        if color_in_hex_format is None:
            color_after:str|None = None
            pass
        else:
            color_after = color_in_hex_format
            pass
        pass
    else:#color is not None
        color_inner = correct_color(color)
        color_after = None
        pass
    fmt = f'{marker_inner}{line_inner}{color_inner}'

    #data
    data_y_inner = any_1d_list_to_pylist(data_y)
    assert data_y_inner.__len__()>0,"At least some data to plot, right?"
    if data_x is None:
        data_x_inner = None
        result = the_axes.plot(data_y_inner, fmt, label = label, 
                        linewidth=linewidth, markersize=markersize, 
                        scalex = scalex, scaley = scaley)[0]
        pass#if 
    else:#data_x is not None
        data_x_inner = any_1d_list_to_pylist(data_x)
        assert data_x_inner.__len__() == data_y_inner.__len__()
        result = the_axes.plot(data_x_inner, data_y_inner, fmt, label = label, 
                        linewidth=linewidth, markersize=markersize, 
                        scalex = scalex, scaley = scaley)[0]
        pass#else
    if color_after is not None:
        result.set_color(color_after)
        pass
    return [result]



def set_legend_according_to_existing_artists()->legend.Legend:
    '''collect all the legend info and add a real legend object into axes.'''
    return plt.legend()

def legend_only_for(handles:list[artist.Artist])->legend.Legend:
    return plt.legend(handles)

def set_legend_temporarily(handles:list[artist.Artist], names:list[str])->legend.Legend:
    '''This function only affect the displayed legend. The stored legend in Artists are not modified.
    
    The unmentioned Artists will be ignored. So, you can ignore some Artists in legend by not adding them
    to the handles list.
    
    For more parameters:https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend
    '''
    for handle in handles:
        assert isinstance(handle, artist.Artist), "I'm not sure, but this list should probably only contains Artists."
        pass
    assert handles.__len__() == names.__len__(), "Check the lists."
    return plt.legend(handles, names)

if "line chart and legend" and __DEBUG_ME__() and False:
    set_style(5)
    
    data_1_x = torch.tensor([1,2])
    data_1_y = torch.tensor([1,3])
    data_2_x = torch.tensor([5,6])
    data_2_y = torch.tensor([5,7])

    line2d_1 = LineChart_on_axes(data_1_x, data_1_y, "data 1111", marker="+", line="--")[0]
    line2d_2 = LineChart_on_axes(data_x=data_2_x, data_y=data_2_y, 
                    label="data 2222", marker="2", line="-", linewidth=5, markersize=15,
                    color_in_hex_format="#CA1616")[0]
    
    set_legend_temporarily([line2d_1], ["new label for 1"])
    
    plt.show()
    #you should see 2 lines and legend with only 1 entry.
    pass






# plt.plot()

# axes.set_xticks(range(len(x_tick_names)), labels=x_tick_names,
#                 rotation=45, ha="right", rotation_mode="anchor")
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html#matplotlib.axes.Axes.set_xticks




# def Contour_fill_on_axes():
#     1w 继续。
#     plt.contourf()



# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# fig, ax = plt.subplots()
# im = ax.imshow(harvest)


def image_show_simple_combined_style_1(data:data_list_2d, title:str="",
            x_tick_names:list[str]|None = None, y_tick_names:list[str]|None = None, 
            vmin: float | None = None,vmax: float | None = None,
            the_axes = get_current_axes())-> tuple[image.AxesImage, list[axis.Tick]|None, list[axis.Tick]|None, 
                                                list[list[text.Text]], text.Text]:
    '''
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    '''
    the_image = the_axes.imshow(data, vmin=vmin, vmax=vmax)

    # Show all ticks and label them with the respective list entries
    if x_tick_names is not None:
        xticks = the_axes.set_xticks(range(len(x_tick_names)), labels=x_tick_names,
                rotation=45, ha="right", rotation_mode="anchor")
        pass
    else:
        xticks = None
        pass
    if y_tick_names is not None:
        yticks = the_axes.set_yticks(range(len(y_tick_names)), labels=y_tick_names)
        pass
    else:
        yticks = None
        pass

    # Loop over data dimensions and create text annotations.
    output_text_list_list:list[list[text.Text]] = []
    for ii in range(data.__len__()):
        output_text_list_list.append([])
        for jj in range(data[0].__len__()):
            _the_text = the_axes.text(jj, ii, f"{data[ii][jj]:.3f}",
                        ha="center", va="center", color="w", rotation=30)
            output_text_list_list[ii].append(_the_text)
            pass#for inner
        pass#for

    the_title_text = the_axes.set_title(title)

    return the_image, xticks, yticks, output_text_list_list, the_title_text

def set_tight_layout(figure:figure.Figure = get_current_figure()):
    figure.tight_layout()



if "image show test" and __DEBUG_ME__() and False:
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    harvest = numpy.array( [[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                            [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                            [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                            [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                            [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                            [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                            [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    title_of_the_chart = "Harvest of local farmers (in tons/year)"
    image_show_simple_combined_style_1(harvest,title_of_the_chart,farmers,vegetables)
    set_tight_layout()

    plt.show()
    pass



#https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contourf.html
if "contour test":

    x = numpy.arange(1, 10)
    y = x.reshape(-1, 1)
    h = x * y


def contour_fill_example_1(data:data_list_2d, the_axes = get_current_axes())-> contour.QuadContourSet:
    '''
    contour.QuadContourSet for more details
    '''
    The_contour_fill = the_axes.contourf(h)
    The_contour_fill.cmap.set_over('red')
    The_contour_fill.cmap.set_under('blue')
    The_contour_fill.changed()
    return The_contour_fill

def contour_fill_example_2(x_info:data_list, y_info:data_list, data:data_list_2d, 
                                the_axes = get_current_axes())-> contour.QuadContourSet:
    '''
    contour.QuadContourSet for more details
    '''
    The_contour_fill = the_axes.contourf(x_info, y_info, h)
    The_contour_fill.cmap.set_over('red')
    The_contour_fill.cmap.set_under('blue')
    The_contour_fill.changed()
    return The_contour_fill

    
def contour_fill_with_manual_level_example_1(data:data_list_2d, level_boundary:data_list, 
                                        level_colors, the_axes = get_current_axes())-> contour.QuadContourSet:
    '''
    contour.QuadContourSet for more details
    '''
    The_contour_fill = the_axes.contourf(h, levels=level_boundary, colors=level_colors, extend='both')
    The_contour_fill.cmap.set_over('red')
    The_contour_fill.cmap.set_under('blue')
    The_contour_fill.changed()
    return The_contour_fill

def contour_fill_with_manual_level_example_2(x_info:data_list, y_info:data_list, data:data_list_2d,
                level_boundary:data_list, level_colors, the_axes = get_current_axes())-> contour.QuadContourSet:
    '''
    contour.QuadContourSet for more details
    '''
    The_contour_fill = the_axes.contourf(x_info, y_info, h, levels=level_boundary, colors=level_colors, extend='both')
    The_contour_fill.cmap.set_over('red')
    The_contour_fill.cmap.set_under('blue')
    The_contour_fill.changed()
    return The_contour_fill



if "color map type's definition" and "to fold the lines":
    #https://matplotlib.org/stable/gallery/color/colormap_reference.html
    #print(matplotlib.colormaps)
    #print(plt.colormaps())
    type color_map_type = Literal['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 
        'twilight_shifted', 'turbo', 'berlin', 'managua', 'vanimo', 'Blues', 'BrBG', 'BuGn',
        'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 
        'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 
        'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary',
        'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 
        'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 
        'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink',
        'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain', 'winter', 'Accent', 
        'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 
        'tab20b', 'tab20c', 'grey', 'gist_grey', 'gist_yerg', 'Grays', 'magma_r', 'inferno_r',
        'plasma_r', 'viridis_r', 'cividis_r', 'twilight_r', 'twilight_shifted_r', 'turbo_r', 
        'berlin_r', 'managua_r', 'vanimo_r', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r',
        'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r',
        'PuBuGn_r', 'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 
        'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r',
        'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r',
        'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 
        'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r',
        'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r', 'pink_r',
        'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 'winter_r',
        'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r',
        'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r', 'grey_r', 'gist_grey_r', 'gist_yerg_r',
        'Grays_r']
    pass

type origin_type = Literal[None, 'upper', 'lower', 'image']

def set_axis_off(the_axes = get_current_axes())->None:
    the_axes.set_axis_off()



# plt.contourf([1,2,3],[6,7],[[11,12,13],[21,22,23]], levels=[5, 15, 25],\
#             corner_mask = True, colors=['#808080', '#A0A0A0', '#C0C0C0'],\
#                 #or single color
#             alpha = 1., #cmap = 'viridis',\
#                 #:matplotlib.colors.Colormap|color_map_type|None = None,
#             norm = None,#str or Normalize, optional
            
#             extend='both'
#         )

#plt.show()

if "contour test" and __DEBUG_ME__() and False:
    pass




#plt.matshow() calls imshow inside. So let me ignore this.




# 还有 hist
# 还有 hist
# 还有 hist
# 还有 hist
# 还有 hist
# 还有 hist
# 还有 hist



#plt.plot(x,y)

# plt.Axes.plot
# plt.Line2D
# plt.gca()
# plt.gca().plot()#list[Line2D]


# plt.gcf()# Figure
# plt.gcf().get_figure()#Return the .Figure or .SubFigure instance the (Sub)Figure belongs to.
# plt.gcf().get_axes()# list[Axes]
# plt.gcf().get_children()# list[Artist]

# plt.gca()# Axes
# plt.gca().get_figure()#Return the .Figure or .SubFigure instance the artist belongs to.
# plt.gca().get_children()# list[Artist]








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

