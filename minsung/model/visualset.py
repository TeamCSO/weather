# Unify visualization style
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import os
# color settings:
def Graphset():
    sns.set(rc={'axes.facecolor':'#ffffff',
                'figure.facecolor':'#E2E2E2',
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

def Graph_temp_heat(data, date:tuple):
    Graphset()
    dt = str(date[0])+"-"+str(date[1])+"-"+str(date[2])
    dt1 = dt + " 00:00"
    dt2 = dt + " 23:59"
    
    data = data[data['Dates'].between(dt1, dt2)]

    fig, ax = plt.subplots(2,1, figsize=(50,30))


    ax[0].plot(data['Dates'],data['heating_temperature_set_up'], label='heat_setup')
    ax[0].plot(data['Dates'],data['ventilation_temperature_control'], label='vent_setup')
    ax[0].plot(data['Dates'],data['in_tmperature'], label='in_temp')
    ax[0].plot(data['Dates'],data['out_tmperature'], label='out_temp')

    ax[1].plot(data['Dates'],data['heat_supply'], label='heat_supply')

    # ax[2].plot(data['Dates'],data['floating_fan'], label='floating_fan')

    for i in range(2):
        ax[i].tick_params(labelsize=30)
        ax[i].legend(loc='center right', prop={'size':40})
    
    ax[0].set_title('Temperature data', fontsize=60, pad=15)
    ax[1].set_title('Heat supply data', fontsize=60, pad=15)

    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    fig.subplots_adjust(top=0.9)
    fig.suptitle(dt,fontsize="80")
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    plt.savefig(dir_path+"/data/image/"+dt)


    