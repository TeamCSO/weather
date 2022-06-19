# Unify visualization style
import seaborn as sns
from matplotlib import pyplot as plt
# color settings:
def Graphset():
    sns.set(rc={'axes.facecolor':'#EBE0BA',
                'figure.facecolor':'#E0D3AF',
                'grid.color':'#E0D3AF',
                'axes.edgecolor':'#424949',
                'axes.labelcolor':'#424949',
                'text.color':'#424949' # color for headlines and sub headlines
                })
    # font size settings
    sns.set_context(rc={'axes.labelsize':15})

    # Times New Roman: (newspaper look)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']