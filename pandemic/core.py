import sys,cv2,os
import numpy as np
from scipy.stats import truncnorm,norm


################################################
#                                              #
#          Generate initial condition          #
#                                              #
################################################


def init_health(prange=[[0,50],[0,50]],vrange=[0.5,5],num=100):
    """
    Generate the initial positions and velocities for the healthy subjects
    
    Input
    ------
    prange: the boundary of the initial healthy subjects reside
    vrange: the range of the velocity
    num: number of subjects
    
    Output
    ------
    tuple: containing position and velocity arrays, both are (2,num) arrays
    """
    # Range of (x,y) and velocity
    xr,yr=np.asarray(prange)
    vrange=np.asarray(vrange)
    
    # Generate positions
    x = np.random.uniform(xr[0],xr[1],size=num)
    y = np.random.uniform(yr[0],yr[1],size=num)
    
    # Generate velocities
    init_v = np.random.uniform(vrange[0],vrange[1],size=num)
    init_theta = np.random.uniform(0,2*np.pi,size=num)
    x_v = init_v*np.cos(init_theta)
    y_v = init_v*np.sin(init_theta)
    
    return np.asarray([x,y]),np.asarray([x_v,y_v])


def init_ill(recov_spec,dead_spec,prange=[[0,50],[0,50]],vrange=[0.5,5],num=1):
    """
    Generate the initial postions and velocities for the ill subjects
    
    Input
    ------
    recov_spec: recovery spec
    dead_spec: death spec
    prange: the boundary of the initial ill subjects reside
    vrange: the range of the velocity
    num: number of subjects
    
    Output
    ------
    tuple: containing the arrays of position, velocity and time stamps of being
           infected, the first two are (2,num) and the last is (num,) array
    """
    # Generate the positions and velocities of the ill subjects from function init_health
    ill_pos,ill_v = init_health(prange,vrange,num)
    # Generate empty array to store the time stamp that will triger the subject to recover
    # or die in the future
    fut_time = np.zeros(num)
    
    # Generate the array of being infected, the initial ills have t_inf = 0
    recov_time = recovery(num,0,recov_spec)
    dead_time = dead(num,0,dead_spec)
    
    # Determine which index will recover, 1 will recover; 0 will die
    fut_stat = (recov_time < dead_time)*1
    # The corresponding time stamp to recover
    fut_time[recov_time<dead_time] = recov_time[recov_time<dead_time]
    # The corresponding time stamp to die
    fut_time[recov_time>dead_time] = dead_time[recov_time>dead_time]
    
    return ill_pos,ill_v,np.array([fut_stat,fut_time])


################################################
#                                              #
#             Renew the conditions             #
#                                              #
################################################


def next_pos_v(pos,vel,box_size,dt):
    """
    Calculate the next position(s) and velocity(s) of the subject(s)
    
    Input
    -----
    pos: positions of n-th subjects, a (2,n) array
    vel: velocities of n-th subjects, a (2,n) array
    box_size: an (2,2) array, the boundary of the simulation box
    dt: the size of simulation time step
    
    Output
    ------
    tuple: containing the arrays of n-th subjects' next positions and velocities
    """
    pos = np.asarray(pos)
    vel = np.asarray(vel)
    box_size=np.asarray(box_size)
    
    # Calculate the next position
    dx = vel*dt
    next_pos = pos+dx
    
    def _get_x_and_v(x,v,box_size):
        """
        Subroutine to find the true positions and velocities if they exceed the boundary
        
        x: the 1-dim next position arrqy given by last_pos + v*dt
        v: the 1-dim velocity array at the position
        box: a list that indicates the min and max of the boundary
        """
        b_min,b_max = box_size
        
        # Where the position array exceed the boundary
        lower = x < b_min
        larger = x > b_max
        
        # Correct the positions when they are exceeding b_min and b_max
        x_min = (2*b_min - x)*lower
        x_max = (2*b_max - x)*larger
        
        # The correct positions and velocities
        x = x_min + x_max + x*(~lower)*(~larger)
        v = (-1)*lower*v + (-1)*larger*v +(~lower)*(~larger)*v
        return x,v
    
    next_pos[0],vel[0] = _get_x_and_v(next_pos[0],vel[0],box_size[0])
    next_pos[1],vel[1] = _get_x_and_v(next_pos[1],vel[1],box_size[1])

    return next_pos,vel


################################################
#                                              #
#                Sampling core                 #
#                                              #
################################################


def infected(ill,health,vh,dt,ill_spec,mask_protect=None):
    """
    Determined which healthy subjects will be infected in this step.
    
    Input
    ------
    ill: ill subjects' positions
    health: healthy subject's positions
    dt: simulation time step
    ill_spec: [r_inf,t_avg,t_std]
    mask_protect: the protectability of wearing a facial mask.
                  Default is None for no mask is wearing. Input value between 0 and 1
    
    Return
    ------
    array: the indices of the healthy subjects' array that are infected in this step
    """
    num_ill = len(ill[0])
    r_inf,t_avg,t_std = ill_spec
    
    if mask_protect is None:
        pass
    elif mask_protect < 1:
        raise ValueError('The mask protectability should be larger than 1')
    else:
        r_inf = r_inf/mask_protect
    
    # Empty array to store which subjects will be infected in this step
    new_ill = np.zeros_like(health[0])
    
    # Probability function
    def _sampling_infected(vh,dt):
        # Number of high probability subjects
        num = len(vh[0])
        # Calculate the speed of each ill subjects
        speed = np.sqrt(np.sum(vh**2,axis=0))
        # Calculate the time of passing through r_inf
        t_pass = r_inf/speed
        # if t_pass is larger than dt, the replace it with dt
        t_pass[t_pass>dt] = dt
        
        # How much time is needed for this subject to be infected
        t_inf = truncnorm.rvs(-t_avg/t_std,np.inf,loc=t_avg,scale=t_std,size=num)
        # If t_pass > t_inf, the subject will be infected because it stays in the infectious zone too long
        return (t_pass > t_inf)*1
    
    for i in range(num_ill):
        # Calculate the distance between the healthy subjects and the ill one
        r = np.sqrt(np.sum((ill[:,i].reshape(2,1)-health)**2,axis=0))
        # Find which healthy subjects' having the distance to the ill is smaller than r_inf
        high_prob_index = np.where(r<r_inf)[0]
        if high_prob_index.size == 0:
            # No one stands in the place where the distance is smaller than r_inf
            pass
        else:
            # Sampling which high_prob subject is infected
            is_ill = _sampling_infected(vh[:,high_prob_index],dt)
            # Set the corresponding index in the new_ill array 1
            new_ill[high_prob_index] = is_ill
            
    return np.where(new_ill==1)[0]


def dead(num,time,death_spec):
    """
    Determined the time stamp that the subject will die
    
    num: how many subjects' death time want to generate
    time: current time stamp
    death_spec: [t_avg,t_std]
    
    Return
    ------
    scalar: the time stamp that the subject will die
    """
    t_avg,t_std = death_spec
    
    # Get the rest hours that the subject can still live
    # I use truncnorm to truncate the random variate between (0,np.inf) 
    rest_time = truncnorm.rvs(a=-t_avg/t_std,b=np.inf,loc=t_avg,scale=t_std,size=num)
    # Mark when time stamp that the subject will die
    dead_time = time + rest_time
    
    return dead_time


def recovery(num,time,recov_spec):
    """
    Determined the time stamp that the subject will recover
    
    num: how many subjects' recovery time want to generate
    time: the current time stamp
    recov_spec: [t_avg,t_std]
    
    Return
    ------
    scalar: the time stamp that the subject will recover
    """
    t_avg,t_std = recov_spec
    
    # Get how many hours that the subject needs to recover
    # I use truncnorm to truncate the random variate between (0,np.inf) 
    need_time = truncnorm.rvs(a=-t_avg/t_std,b=np.inf,loc=t_avg,scale=t_std,size=num)
    # Mark when time stamp that the subject will die
    recov_time = time + need_time
    
    return recov_time


def sub_stat(ill,health,recov,death,
             ill_v,health_v,recov_v,
             t_ill,
             dt,current_time,
             ill_spec,
             recov_spec,
             death_spec,
             mask_protect=None):
    """
    Update the status of the infected and non-infected subject at this interval
    
    Input
    ------
    ill: the ill subjects' positions
    health: the healthy subjects' positions
    recov: the recovered subjects' positions
    death: the dead subjects' positions
    ill_v: the ill subjects' velocities
    health_v: the healthy subjects' velocities
    recov_v: the recovered subjects' velocities
    t_ill: the time stamp of the ill subjects when they were being infected
    dt: the size of the simulation time step
    current_time: current time stamp
    ill_spec: the specs of getting infected [r_inf,t_inf_avg,t_inf_std]
    recov_spec: the specs of getting recovered [t_r_avg,t_r_std]
    death_spec: the specs of dying [t_d_avg,t_d_std]
    mask_protect: the protectability of wearing a mask
    
    Output
    ------
    The same as input but with updated status
    """
    # Check the t_ill that which one's future are destined to die
    who_will_die = np.where(t_ill[0]==0)[0]
    # Create a list to store the subject index that is going to die in this step
    who_is_dead = []
    if who_will_die.size == 0:
        # no one will die
        pass
    else:
        for sub in who_will_die:
            if current_time > t_ill[1,sub]:
                # this subject's times up
                new_dead = ill[:,sub].reshape(2,1)
                # add to the death list
                death = np.append(death,new_dead,axis=1)
                # store the really dead subject's index
                who_is_dead.append(sub)
            else: pass
        # Remove the dead from the infected positions, velocities and time arrays
        if np.array(who_is_dead).size == 0:
            pass
        else:
            who_is_dead = np.array(who_is_dead)
            ill = np.delete(ill,who_is_dead,axis=1)
            ill_v = np.delete(ill_v,who_is_dead,axis=1)
            t_ill = np.delete(t_ill,who_is_dead,axis=1)
    
    # Check the t_ill that which one's future are destined to recover
    who_will_recover = np.where(t_ill[0]==1)[0]
    # Create a list to store the subject index that is going to recover in this step
    who_is_recovered = []
    if who_will_recover.size == 0:
        # no one will die
        pass
    else:
        for sub in who_will_recover:
            if current_time > t_ill[1,sub]:
                # this subject's times up
                new_recov = ill[:,sub].reshape(2,1)
                new_recov_v = ill_v[:,sub].reshape(2,1)
                # add to the recover list
                recov = np.append(recov,new_recov,axis=1)
                recov_v = np.append(recov_v,new_recov_v,axis=1) 
                # store the really recovered subject's index
                who_is_recovered.append(sub)
            else: pass
        # Remove the recovered from the infected positions, velocities and time arrays
        if np.array(who_is_recovered).size == 0:
            pass
        else:
            who_is_recovered = np.array(who_is_recovered)
            ill = np.delete(ill,who_is_recovered,axis=1)
            ill_v = np.delete(ill_v,who_is_recovered,axis=1)
            t_ill = np.delete(t_ill,who_is_recovered,axis=1)
    
    # Find the array index of the non_inf is infected in this step
    get_ill = infected(ill,health,health_v,dt,ill_spec,mask_protect)
    # Deal with the new_infect
    if get_ill.size == 0:
        # no subject is infected in this interval, pass
        pass
    else:
        # Get the number of new infected
        new_ill_num = len(get_ill)
        
        # Get the pos and v of the new infected
        new_ill = health[:,get_ill]
        new_ill_v = health_v[:,get_ill]
        
        # Append to the ill array
        ill = np.append(ill,new_ill,axis=1)
        ill_v = np.append(ill_v,new_ill_v,axis=1)
        
        # Time of recovery and die
        recov_time = recovery(new_ill_num,current_time,recov_spec)
        dead_time = dead(new_ill_num,current_time,death_spec)
        
        # Determine which index will recover, 1 will recover; 0 will die
        fut_time = np.zeros(new_ill_num)
        fut_stat = (recov_time < dead_time)*1
        # The corresponding time stamp to recover
        fut_time[recov_time<dead_time] = recov_time[recov_time<dead_time]
        # The corresponding time stamp to die
        fut_time[recov_time>dead_time] = dead_time[recov_time>dead_time]
        
        # Append to t_ill
        t_ill = np.append(t_ill,np.array([fut_stat,fut_time]),axis=1)
             
        # Now delete all the new infected ones from the non-infected list (both positions and velocities)
        health = np.delete(health,get_ill,axis=1)
        health_v = np.delete(health_v,get_ill,axis=1)

    return ill,health,recov,death,ill_v,health_v,recov_v,t_ill


################################################
#                                              #
#                Subject class                 #
#                                              #
################################################


class Subject:
    
    def __init__(self,n_ill,n_health,prange,vrange,box_size,inf_spec,recov_spec,dead_spec,dt):
        self.n_ill = n_ill
        self.n_health = n_health
        self.prange = prange
        self.vrange = vrange
        self.box_size = box_size
        self.inf_spec = inf_spec
        self.recov_spec = recov_spec
        self.dead_spec = dead_spec
        # the statistic of the current step
        self._statistic = {'Ill':[],
                           'Health':[],
                           'Recovered':[],
                           'Dead':[],
                           'Time':[],
                           'dt':dt}
        # Raw data of all subjects' status including the preceeding steps
        self._fullout = []
    
    def _update(self,xi,xh,xr,xd,vi,vh,vr,ti,time):
        # update statistic
        self._statistic['Ill'] = np.append(self._statistic['Ill'],len(xi[0]))
        self._statistic['Health'] = np.append(self._statistic['Health'],len(xh[0]))
        self._statistic['Recovered'] = np.append(self._statistic['Recovered'],len(xr[0]))
        self._statistic['Dead'] = np.append(self._statistic['Dead'],len(xd[0]))
        self._statistic['Time']=np.append(self._statistic['Time'],time)
        # Recorde full output at specific time
        self._fullout.append({'IllPosition':xi,
                                 'IllVelocity':vi,
                                 'HealthPosition':xh,
                                 'HealthVelocity':vh,
                                 'RecoveredPosition':xr,
                                 'RecoveredVelocity':vr,
                                 'DeadPosition':xd
                             })
    
    def get_init(self):
        xi,vi,ti = init_ill(recov_spec=self.recov_spec,dead_spec=self.dead_spec,
                            prange=self.prange,vrange=self.vrange,num=self.n_ill)
        xh,vh = init_health(prange=self.prange,vrange=self.vrange,num=self.n_health)
        xr = np.array([[],[]])
        vr = np.array([[],[]])
        xd = np.array([[],[]])
        self._update(xi,xh,xr,xd,vi,vh,vr,ti,0)
        return xi,xh,xr,xd,vi,vh,vr,ti
    
    def run(self,xi,xh,xr,xd,vi,vh,vr,ti,dt,time,mask):
        xi,xh,xr,xd,vi,vh,vr,ti = sub_stat(xi,xh,xr,xd,vi,vh,vr,ti,dt,time,
                                           self.inf_spec,self.recov_spec,self.dead_spec,mask)
        # Update positions and velocities
        xi,vi=next_pos_v(xi,vi,box_size=self.box_size,dt=dt)
        xh,vh=next_pos_v(xh,vh,box_size=self.box_size,dt=dt)
        xr,vr=next_pos_v(xr,vr,box_size=self.box_size,dt=dt)
        self._update(xi,xh,xr,xd,vi,vh,vr,ti,time)
        return xi,xh,xr,xd,vi,vh,vr,ti
    
    @property
    def statistic(self):
        return self._statistic
    
    @property
    def fullout(self):
        return self._fullout
    
    @property
    def init_spec(self):
        # calculate box size
        width = self.box_size[0,1]-self.box_size[0,0]
        height = self.box_size[1,1]-self.box_size[1,0]
        print('Initial health: %d'%self.n_health)
        print('Initial ill   : %d'%self.n_ill)
        print('Box size      : W'+str(self.box_size))
        