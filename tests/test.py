# content of test_sample.py

# Import libraries
import matplotlib
matplotlib.use('Agg')
import base64
import json
import pandas               as pd
from io                 import StringIO,BytesIO
from matplotlib         import pyplot as plt
from pandas.plotting import scatter_matrix

def plot(plot, name):     
    image = BytesIO()    
    fig = plot.get_figure()                   
    fig.savefig(name, format='png')
    # image.seek(0)  # rewind to beginning of file
    # figdata_png = image.getvalue()     
    # d = base64.encodestring(image.getvalue()).decode('ascii')   
    # return '%s' % (d)

def analyse():
    data = pd.read_csv("./test.csv")    
    data.plot()
    # plt.show()            
    sm = scatter_matrix(data, figsize=(24,24))    
    n = 0
    print(plot(sm[0][0], name="./" + str(n) + ".png"))        
            

analyse()
def test_answer():    
    assert 0==0
