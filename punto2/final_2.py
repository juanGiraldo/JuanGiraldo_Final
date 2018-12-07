import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('datos_observacionales.dat')
t,xyz=data[:,0],data[:,1:]

#calcula el loglike, llama a modelo. Se asume una likelihood en forma de multiplicatoria-->log es una suma
#sig = sigma de los datos (error bars)
def model(x,param):
    """Modelo polinomial. `param` contiene los coeficientes.
    """
    n_param = len(param)
    #print(x.shape)
    v=np.zeros(x.shape)
    v[:,0] = param[0] * (x[:,1]-x[:,0])
    v[:,1] = x[:,0]*(param[1]-x[:,2])-x[:,1]
    v[:,2] = x[:,0]*x[:,1]-param[2]*x[:,2]
    return v
    
def loglikelihood(x_obs, y_obs, sigma_y_obs, param):
    """Logaritmo natural de la verosimilitud construida con los datos observacionales y los 
        parametros que describen el modelo.
    """
    d = y_obs -  model(y_obs, param)
    d[:,0]=d[:,0]/sigma_y_obs
    d[:,1]=d[:,1]/sigma_y_obs
    d[:,2]=d[:,2]/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

def logprior(param):
    """Logaritmo natural de los prior para los parametros.
        Todos corresponden a gaussianas con sigma=10.0.
    """
    d = -0.5 * np.sum(param**2/(10.0)**2)
    return d

def divergence_loglikelihood(x_obs, y_obs, sigma_y_obs, param):
    """Divergencia del logaritmo de la funcion de verosimilitud.
    """
    n_param = len(param)
    div = np.ones(n_param)
    delta = 1E-5
    for i in range(n_param):
        delta_parameter = np.zeros(n_param)
        delta_parameter[i] = delta
        div[i] = loglikelihood(x_obs, y_obs, sigma_y_obs, param + delta_parameter) 
        div[i] = div[i] - loglikelihood(x_obs, y_obs, sigma_y_obs, param - delta_parameter)
        div[i] = div[i]/(2.0 * delta)
    return div

def hamiltonian(x_obs, y_obs, sigma_y_obs, param, param_momentum):
    """Hamiltoniano: energia cinetica + potencial: K+V
    """
    m = 100.0
    K = 0.5 * np.sum(param_momentum**2)/m
    V = -loglikelihood(x_obs, y_obs, sigma_y_obs, param)     
    return K + V

def leapfrog_proposal(x_obs, y_obs, sigma_y_obs, param, param_momentum):
    """Integracion tipo leapfrog. 
        `param` representa las posiciones (i.e. los parametros).
        `param_momemtum` representa el momentum asociado a los parametros.
    """
    N_steps = 5
    delta_t = 1E-2
    m = 100.0
    new_param = param.copy()
    new_param_momentum = param_momentum.copy()
    for i in range(N_steps):
        new_param_momentum = new_param_momentum + divergence_loglikelihood(x_obs, y_obs, sigma_y_obs, param) * 0.5 * delta_t
        new_param = new_param + (new_param_momentum/m) * delta_t
        new_param_momentum = new_param_momentum + divergence_loglikelihood(x_obs, y_obs, sigma_y_obs, param) * 0.5 * delta_t
    new_param_momentum = -new_param_momentum
    return new_param, new_param_momentum


def monte_carlo(x_obs, y_obs, sigma_y_obs, N=5000):
    param = [np.random.random(3)]
    param_momentum = [np.random.normal(size=3)]
    for i in range(1,N):
        propuesta_param, propuesta_param_momentum = leapfrog_proposal(x_obs, y_obs, sigma_y_obs, param[i-1], param_momentum[i-1])
        energy_new = hamiltonian(x_obs, y_obs, sigma_y_obs, propuesta_param, propuesta_param_momentum)
        energy_old = hamiltonian(x_obs, y_obs, sigma_y_obs, param[i-1], param_momentum[i-1])
   
        r = min(1,np.exp(-(energy_new - energy_old)))
        alpha = np.random.random()
        if(alpha<r):
            param.append(propuesta_param)
        else:
            param.append(param[i-1])
        param_momentum.append(np.random.normal(size=3))    

    param = np.array(param)
    return param
    
param_chain = monte_carlo(t, xyz, err)
n_param  = len(param_chain[0])
best = []
for i in range(n_param):
    best.append(np.mean(param_chain[:,i]))
    
sigma=param_chain[:,0]
ro =  param_chain[:,1]
beta= param_chain[:,2]

bins = np.linspace(0, 0.7, 1000)

plt.hist(sigma, bins, alpha=0.5, label='$\sigma$')
plt.hist(ro, bins, alpha=0.5, label='rho')
plt.hist(beta, bins, alpha=0.5, label='beta')
plt.legend(loc='upper right')
plt.savefig()