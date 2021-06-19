import os,sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count
#mp.get_start_method('spawn')
from timeit import default_timer as timer
from pandsim import loadsim
import seaborn as sns
sns.set(color_codes=True)


################################################
#                                              #
#         Multiprocessing for drawsim          #
#           *!*!* EXPERIMENTAL *!*!*           #
#                                              #
################################################


def _drawfullout(s,out,info,disease_name,box_size,ext_x,ext_y,skip,dpi):
    out = out
    info = info
    fig = plt.figure()
    plt.scatter(out[s]['HealthPosition'][0],out[s]['HealthPosition'][1],c='limegreen',alpha=0.6)
    plt.scatter(out[s]['IllPosition'][0],out[s]['IllPosition'][1],c='orangered',alpha=0.6)
    plt.scatter(out[s]['RecoveredPosition'][0],out[s]['RecoveredPosition'][1],c='steelblue',alpha=0.6)
    plt.scatter(out[s]['DeadPosition'][0],out[s]['DeadPosition'][1],c='k',alpha=0.6)
    plt.xlim(box_size[0,0]-ext_x,box_size[0,1]+ext_x)
    plt.ylim(box_size[1,0]-ext_y,box_size[1,1]+ext_y)
    plt.title(r'$t=$'+str(int(info['Time'][s]//24))+'d'
                +r', $N_{\rm h}=$'+str(int(info['Health'][s]))
                +r', $N_{\rm ill}$='+str(int(info['Ill'][s]))
                +r', $N_{\rm recov}=$'+str(int(info['Recovered'][s]))
                +r', $N_{\rm dead}=$'+str(int(info['Dead'][s])))
    plt.xlabel('$X$ [meters]')
    plt.ylabel('$Y$ [meters]')
    plt.savefig(str(disease_name)+'/images/'+str(s//skip+1)+'.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def drawsim_mp(disease_name,skip=None,dpi=150,cores=int(cpu_count()/2)):
    '''
    Drawing figures
    
    Input
    ------
    disease_name: name of the project folder in the same location
    skip: how many dt should be skip between figures
    dpi: resolution
    cores: how many number of cores to be parallized, default is half of the machine cores
    
    Output
    ------
    PNG figures of specified steps and summary plot
    '''
    # check if images folder exists, if not, create it
    if os.path.isdir(str(disease_name)+'/images'):
        pass
    else:
        os.mkdir(str(disease_name)+'/images')
    
    # load data
    info,out = loadsim(disease_name)
    # setup
    box_size = np.asarray(info['BoxSize'])
    time_stamp = info['Time']
    steps = len(time_stamp)
    # extension on the boundary when drawing
    ext_x = (box_size[0,1]-box_size[0,0])*0.025
    ext_y = (box_size[1,1]-box_size[1,0])*0.025
    
    if type(skip) == int and skip > 0:
        label_range = range(0,steps,skip)
        #steps_label = np.ceil(steps/skip)
    elif skip is None:
        label_range = range(steps)
        #steps_label = steps
        skip = 1
    else:
        raise ValueError('The input skip is not a positive integer, please check again')
    
    start = timer()
    # drawing summary plot
    fig = plt.figure()
    plt.plot(info['Time']/24,info['Health'],label='health',c='limegreen')
    plt.fill_between(info['Time']/24, info['Health'],color='limegreen',alpha=0.4)
    plt.plot(info['Time']/24,info['Ill'],label='ill',c='orangered')
    plt.fill_between(info['Time']/24,info['Ill'],color='orangered',alpha=0.4)
    plt.plot(info['Time']/24,info['Recovered'],label='recovered',c='steelblue')
    plt.fill_between(info['Time']/24, info['Recovered'],color='steelblue',alpha=0.4)
    plt.plot(info['Time']/24,info['Dead'],label='dead',c='k')
    plt.fill_between(info['Time']/24, info['Dead'],color='k',alpha=0.4)
    plt.title('Time evolving statistics of '+str(disease_name))
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.savefig(str(disease_name)+'/images/summary.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    
    print('Parallizing the drawing process, please wait...',end='\r')
    # Multiprocessing the drawing process
    with Pool(cores) as pool:
        for s in label_range:
            pool.apply_async(_drawfullout,(s,out,info,disease_name,box_size,ext_x,ext_y,skip,dpi))
        pool.close()
        pool.join()
    sys.stdout.flush()
    end = timer()
    
    # Create and save log file
    img_array = []
    for i in range(1,len(label_range)+1):
        img_array.append(str(i)+'.png')
    np.savetxt(str(disease_name)+'/images/imginfo.txt',np.asarray(img_array),fmt='%s')
    
    print('Drawing process has completed in '+str(np.round(end-start,2))+' seconds.\nLog file imginfo.txt saved!')

    
