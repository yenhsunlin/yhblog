import numpy as np
from scipy.signal import correlate2d


gpu_flag = False

def energy(state,j,h,window):
    """
    Compute the energies of atoms on the grid
    
    Input
    ------
    state: 2D array, current spin configuration of the
        state, +1 for spin-up and -1 for spin-down
    j: coupling constant
        positive is ferromagnetism
        negative is anti-ferromagnetism
        zero means no correlation
    h: spin tendency, contribution from external force
        positive means the spin aligning the ext. force
        negative means the spin anti-aligning the ext. force
        zero means no external force
    window: a 2D window with how many neighbors should
        be considered to calculate the spin sum
    
    Output
    ------
    E: the energy
    """
    # sum of neighborhood spins
    spin_sum = correlate2d(state,window,mode='same')
    E = -j*spin_sum*state - h*state
    return E


def MH_sampling(state,j,h,beta,window):
    """
    Metropolis-Hastings algorithm for determining spin
    configuration in the next step
    
    Input
    ------
    state: 2D array, current spin configuration of the
        state, +1 for spin-up and -1 for spin-down
    j: coupling constant
        positive is ferromagnetism
        negative is anti-ferromagnetism
        zero means no correlation
    h: spin tendency, contribution from external force
        positive means the spin aligning the ext. force
        negative means the spin anti-aligning the ext. force
        zero means no external force
    beta: temperature related, positively defined
    window: a 2D window with how many neighbors should
        be considered to calculate the spin sum
        
    Output
    ------
    next_state: the spin configuration of the next
        step given by Metropolis sampling
    """
    # check if beta is positively defined
    if beta < 0:
        raise ValueError('beta should be postively defined')
    else: pass
    
    new_state = np.zeros_like(state)
    
    # MH SOP step 1
    E = energy(state=state,j=j,h=h,window=window)
    deltaE = -2*E
    
    # MH SOP step 2 
    new_state[deltaE < 0] = -state[deltaE < 0]
    
    # MH SOP step 3 
    # find where has delta E >= 0
    x_plus,y_plus = np.where(deltaE >= 0)
    # calculate the flipping probability
    p_flip = np.exp(-beta*deltaE[x_plus,y_plus])
    # sampling which element should flip
    rnd = np.random.uniform(size=p_flip.shape)
    flip_pos = -1*(rnd < p_flip)
    # those no need to flip
    flip_pos[flip_pos == 0] = 1
    # update state
    new_state[x_plus,y_plus] = state[x_plus,y_plus]*flip_pos
    
    return new_state


def spinMH(size=[500,500],p0=[0.5,0.5],j=1,h=0,beta=1,iters=100,    \
           seed=None,window=np.array([[1,1,1],                      \
                                      [1,0,1],                      \
                                      [1,1,1]])                     \
          ):
    """
    Using Metropolis-Hastings algorithm to sample the
    evolution of the spin configuration of a given initial
    state with j, h and beta
    
    Input
    ------
    size: default [500,500] the size of the grid
    p0: default [0.5,0.5], the probability of spin is being
        up or down in the initial state, total cannot exceed 1
    j: coupling constant
        positive is ferromagnetism
        negative is anti-ferromagnetism
        zero means no correlation
    h: spin tendency, contribution from external force
        positive means the spin aligning the ext. force
        negative means the spin anti-aligning the ext. force
        zero means no external force
    beta: temperature related, positively defined
    iters: default 1000, how many interations should proceed 
    seed: the seed for grenerating the initial state 
    window: a 2D window with how many neighbors should
        be considered to calculate the spin sum
        
    Output
    ------
    ini: the initial spin configuration
    final: the final spin configuration
    """
    window = np.asarray(window)
    np.random.seed(seed)
    ini_state = np.random.choice([1,-1],size=(size[0],size[1]),p=p0)
    new_state = np.copy(ini_state)
    
    for i in range(iters):
        new_state= MH_sampling(new_state,j,h,beta,window)
    
    return ini_state,new_state