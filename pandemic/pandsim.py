import sys,cv2,os
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import seaborn as sns
sns.set(color_codes=True)
from core import Subject


################################################
#                                              #
#                Main function                 #
#                                              #
################################################


def PandemicSimulation(n_ill,n_health,
                       inf_spec=[1,0.25,0.5],recov_spec=[35*24,10*24],
                       dead_spec=[40*24,10*24],mask_protect=None,
                       prange=[[-250,250],[-250,250]],vrange=[5,30],
                       box_size=[[-600,600],[-600,600]],dt=0.5,steps=24*30*2,
                       save_data=False,disease_name='ukn_disease',
                       self_adaptive=False,dpi=150):
    """
    This function runs the simulation of pandemic spreading with the given specs
    
    Input
    ------
    n_ill : number of ill subject in the beginning
    n_health: number of healthy subject in the beginning
    inf_spec: infectious spec
    recov_spec: recovery spec
    dead_spec: death spec
    mask_protect: mask protectability
    prange: the initial position range of the subjects to be generated
    vrange: the velocity range of the subjects to be generated
    box_size: the size of the simulation box
    dt: time slice of each step
    steps: total steps to be run
    disease_name: naming the name of the disease in this simulation
    save_data: save the simulated data, the name will be given by disease_name
    
    Output
    ------
    dict: dictionary that stores the statistics 
    """
    # Turn on self-adaptive adjustment on dt and steps?
    if self_adaptive:
        dt,steps = _self_adaptive_dt(dt,steps,vrange,inf_spec[0],mask_protect)
    else:
        pass
    
    # check if initial position exceeds the box_size
    prange = _checkbox(prange,box_size)
    sub = Subject(n_ill,n_health,prange,vrange,box_size,inf_spec,recov_spec,dead_spec,dt)
    # Get the initial condition and setting time stamp
    xi,xh,xr,xd,vi,vh,vr,ti = sub.get_init()
    time = 0
    
    # Run the simulation until the maximum step is reached
    start_time = timer()
    for s in range(steps):
        time += dt
        xi,xh,xr,xd,vi,vh,vr,ti = sub.run(xi,xh,xr,xd,vi,vh,vr,ti,dt,time,mask_protect)
        #print(str(s)+' out of '+str(steps)+' steps are completed',end='\r')
        print('Progress: '+'%.1f%% completed'%(100*s/steps),end='\r')
        sys.stdout.flush()
    end_time = timer()
    print('Simulation with total %d steps completed in %.3f seconds'%(steps,end_time - start_time))
    
    def _save(disease_name,sim_out):
        # check if folder exists, if not, create one
        if os.path.isdir(str(disease_name)):
            pass
        else:
            os.mkdir(str(disease_name))
        
        np.save(str(disease_name)+'/'+str(disease_name)+'_fullout.npy', sim_out.fullout)
        # Append the following to the summary dict
        summary = sim_out.statistic
        summary['BoxSize'] = np.asarray(box_size)
        summary['InfectedSpec'] = np.asarray(inf_spec)
        summary['RecovereddSpec'] = np.asarray(recov_spec)
        summary['DeadSpec'] = np.asarray(dead_spec)
        summary['Mask'] = mask_protect
        np.save(str(disease_name)+'/'+str(disease_name)+'_summary.npy', summary)
    
    if save_data:
        _save(disease_name,sub)
        print('Data saved.')
    else: pass
    
    # Plot statistics
    plt.plot(sub.statistic['Time']/24,sub.statistic['Health'],label='health',c='limegreen')
    plt.fill_between(sub.statistic['Time']/24, sub.statistic['Health'],color='limegreen',alpha=0.4)
    plt.plot(sub.statistic['Time']/24,sub.statistic['Ill'],label='ill',c='orangered')
    plt.fill_between(sub.statistic['Time']/24,sub.statistic['Ill'],color='orangered',alpha=0.4)
    plt.plot(sub.statistic['Time']/24,sub.statistic['Recovered'],label='recovered',c='steelblue')
    plt.fill_between(sub.statistic['Time']/24, sub.statistic['Recovered'],color='steelblue',alpha=0.4)
    plt.plot(sub.statistic['Time']/24,sub.statistic['Dead'],label='dead',c='k')
    plt.fill_between(sub.statistic['Time']/24, sub.statistic['Dead'],color='k',alpha=0.4)
    plt.title('Time evolving statistics of '+ str(disease_name))
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.savefig(str(disease_name)+'/'+str(disease_name)+'_summary.png',dpi=dpi,bbox_inches='tight')
    
    return sub.statistic


################################################
#                                              #
#                Standard I/O                  #
#                                              #
################################################


def loadsim(disease_name):
    '''
    Load the simulation data
    '''
    summary = np.load(str(disease_name)+'/'+str(disease_name)+'_summary.npy',allow_pickle=True).item()
    fullout = np.load(str(disease_name)+'/'+str(disease_name)+'_fullout.npy',allow_pickle=True)
    return summary,fullout


def drawsim(disease_name,skip=None,dpi=150):
    '''
    Drawing figures
    
    Input
    ------
    disease_name: name of the project folder in the same location
    skip: how many dt should be skip between figures
    
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
        steps_label = np.ceil(steps/skip)
    elif skip is None:
        label_range = range(steps)
        steps_label = steps
        skip = 1
    else:
        raise ValueError('The input skip is not a positive integer, please check again')
    
    start = timer()

    # drawing each step
    img_array =[]
    for s in label_range:
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
        img_array.append(str(s//skip+1)+'.png')
        print(str(s//skip+1)+' out of '+str(int(steps_label))+' are plotted',end='\r')
        sys.stdout.flush()
    end = timer()
    # Save img info for later video making
    np.savetxt(str(disease_name)+'/images/imginfo.txt',np.asarray(img_array),fmt='%s')
    print('Drawing process has completed in '+str(np.round(end-start,2))+' seconds.\nLog file imginfo.txt saved!')

    
def mkvideo(disease_name,fps=15):
    """
    Making video from simulation data
    """
    # load image info
    img_ls = list(np.loadtxt(str(disease_name)+'/images/imginfo.txt',dtype=str))
    for img in range(len(img_ls)):
        img_ls[img] = str(disease_name)+'/images/'+img_ls[img]
    
    # check if video folder exists, if not, create it
    if os.path.isdir(str(disease_name)+'/video'):
        pass
    else:
        os.mkdir(str(disease_name)+'/video')
    
    img_array = []
    for filename in img_ls:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter(str(disease_name)+'/video/'+str(disease_name)+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


################################################
#                                              #
#             Auxiliary functions              #
#                                              #
################################################


def _checkbox(prange,box_size):
    """
    Check if the prange lies outside the box_size.
    If true, the renturn chop the exceeding range
    """
    prange = np.asarray(prange)
    box_size = np.asarray(box_size)
    
    # Does the lower limit exceeds the box boundary
    check_low = prange[:,0] > box_size[:,0]
    prange[:,0] = prange[:,0]*check_low + box_size[:,0]*(~check_low)
    # Does the upper limit exceeds the box boundary
    check_high = prange[:,1] < box_size[:,1]
    prange[:,1] = prange[:,1]*check_high + box_size[:,1]*(~check_high)
    
    return prange


def _self_adaptive_dt(dt,steps,vrange,r_inf,mask_protect):
    """
    Self adjust the dt to fit the resolution of such simulation system
    """
    if mask_protect is None:
        mask_protect = 1
    else: pass
    r_inf = r_inf/mask_protect
    tot_time = dt*steps
    # average velocity
    v_avg = np.sum(vrange)/2
    # does the current dt is able to resolve r_inf
    if v_avg*dt > r_inf:
        # unable to resolve, re-calculate dt and total steps
        new_dt = 0.5*r_inf/v_avg
        new_steps = int(tot_time/new_dt)+1
        print('Self-adaptive dt is on, the new dt and steps are '+str(np.round(new_dt,3))+' and ' +str(new_steps)+'.')
    else:
        # able to resolve, do nothing
        new_dt = dt
        new_steps = steps
        print('Self-adaptive dt is on, the original dt is able to resolve the simulation system.')
    return new_dt,new_steps