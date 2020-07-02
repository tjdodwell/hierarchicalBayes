from fenics import *

from numpy import where

import numpy as np

import matplotlib.pyplot as plt
import random as rnd
###############################################################################
from smt.sampling_methods import LHS
from emukit.model_wrappers.gpy_quadrature_wrappers import BaseGaussianProcessGPy, RBFGPy
from emukit.quadrature.kernels.integration_measures import IsotropicGaussianMeasure
from emukit.quadrature.kernels import QuadratureRBFIsoGaussMeasure
from emukit.quadrature.methods.vanilla_bq import VanillaBayesianQuadrature
import GPy
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import math
from emukit.core.loop.user_function import UserFunctionWrapper
import random as rnd
import seaborn as sns
from typing import Tuple, List, Optional
from scipy.stats import norm
import Gen_data3





# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


class HyperElastic():
    
    def __init__(self,data):#,loadSteps, delta, verbosity, saveSolution):
        
        self.data = data
        self.J = len(data)
        self.N_MC = 30
        self.sigmaf = 0.5

        #self.loadSteps = loadSteps
        #self.delta = delta
        
        # Create mesh and define function space
        self.mesh = UnitCubeMesh(9, 9, 12)
        #mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(1., 4., 2.), 10, 10, 10)
        self.V = VectorFunctionSpace(self.mesh, "Lagrange", 1)
        
        # Mark boundary subdomians
        self.left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
        self.right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

        self.top =  CompiledSubDomain("near(x[2], side) && on_boundary", side = 1.0)
        self.bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0)
        self.c = Constant((0.0, 0.0, 0.0))
        self.bc_bottom = DirichletBC(self.V, self.c, self.bottom) #boundary for bottom face = fixed
        # Find dofs on top face
        ff = MeshFunction("size_t",self.mesh, self.mesh.topology().dim()-1, 0)
        self.top.mark(ff, 1)
        vv = Function(self.V)
        bc = DirichletBC(self.V, Constant((0.0, 0.0,1.0)), ff, 1)
        bc.apply(vv.vector())
        self.top_dofs = where(vv.vector()==1.0)[0]

    def apply(self,data,theta, delta, loadSteps, verbosity = False, saveSolution = False):
                
        # Define functions
        du = TrialFunction(self.V)            # Incremental displacement
        v  = TestFunction(self.V)             # Test function
        u  = Function(self.V)                 # Displacement from previous iteration
        B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
        T  = Constant((0.0,  0.0, 0.0))  # Traction force on the boundary
        
        # Kinematics
        d = u.geometric_dimension()
        I = Identity(d)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor
        
        Cinv = inv(C)
        
        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)
        
        # Composite part
        
        e0 = Constant((1.0, 0.0, 0.0))
        n0 = Constant((0.0, 0.0, 1.0))
        
        L = dot(e0,C * e0)
        calL = dot(n0, Cinv * n0)
        
        # Invariants of deformation tensors
        I1 = tr(C) # This is the first invariant of C
        I2 = 0.5*(tr(C)**2 - tr(dot(C,C)))
        J  = det(F)
        I1_bar = J**(-2/3)*I1
        I2_bar = J**(-4/3)*I2
        ####################################################
        # Elasticity parameters
        #E, nu = 0.000001, 0.3        
        #G, K = E/(2*(1 + nu)), E*nu/((1 + nu)*(1 - 2*nu))

        #theta_0 = Constant(theta[0])
        #theta_1 = Constant(theta[1])
        #Data_Driven_Term = theta_0 * ((1.0 - calL)**2 + theta_1 * (1.0 - calL)**4)

        #Es = Constant(120.0)
        #phi0 = Constant(0.45)
        #dWdl = 0.5 * As * (calL - sqrt(calL))/((1/phi0 - sqrt(calL)/phia)**4) + 0.5 * G * (1 / calL**2 - 1/calL)

        # Stored strain energy density (compressible neo-Hookean model)
        #psi = (G/2)*(I1 - 3) - G*ln(J) + (K/2)*(ln(J))**2 + 0.25 * Es * phi0 * (L - ln(L)) + Data_Driven_Term

        
        
        
        ####################################################
        # Elasticity parameters
        E, nu = 10.0, 0.3
        mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
        
        # Stored strain energy density (compressible neo-Hookean model)
        
        #theta = [10.0, 1.0]
        normal_compaction_model = 0.5 * theta[0] * (calL - ln(calL))
        
        Es, phi0 = 120., 0.45
        axial_model = 0.25 * Es * phi0 * (L - ln(L))
        
        psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2 + normal_compaction_model+axial_model
        ####################################################
        W=0
        #C_10 = theta[0]
        #C_01 = theta[1]
        
        
        #W = C_01*(I2_bar-3) + C_10*(I1_bar - 3)

        # Total potential energy
        #Pi = W*dx - dot(B, u)*dx - dot(T, u)*ds
        Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = derivative(Pi, u, v)
        
        # Compute Jacobian of F
        J = derivative(F, u, du)
        
        
        # Loading loop
        count=0
        misfit = np.zeros(loadSteps)
        for i in range(1, loadSteps + 1):
        
                    if(verbosity):
                        print("Load step - " + str(i))
        
                    # Update non-homogeneous boundary condition for current load step
                    r = Constant((i * delta, 0.0, 0.0))
                    bc_top = DirichletBC(self.V, r, self.top)
                    bcs = [self.bc_bottom, bc_top]
        
                    # Solve variational problem
                    solve(F == 0, u, bcs, J=J)
        
                    # Save solution in VTK format
                    if(saveSolution):
                        file = File("displacement_" + str(i) + ".pvd");
                        file << u;
        
                    # Output forces
                    y = assemble(F)
                    Force_top = 0
                    for i in self.top_dofs:
                        Force_top += y[i]
        
                    print(Force_top)
                    misfit[count] = (data[count] - Force_top)
                    count=count+1

        Phi=self.evaluateDensity(misfit)       #number 0 is nothing 
        return np.exp(Phi)
        
    def evaluateDensity(self, misfit):
        return -0.5 * np.sum(misfit)**2 / (self.sigmaf**2)
##############################################################################################################
    def press_fem(self) -> Tuple[UserFunctionWrapper, List[Tuple[float, float]]]:
        
        integral_bound = [(1,15),(0.2,0.4)]
        return UserFunctionWrapper(self._press_fem), integral_bound

    def _press_fem(self, data,param:np.ndarray, delta,loadSteps, verbosity, saveSolution) -> np.ndarray:
        Likelihood=self.apply(data, param, delta,loadSteps, verbosity, saveSolution)
        return np.reshape(np.array(Likelihood),(-1,1))
##############################################################################################################
    def singleGP(self, data, delta,loadSteps, verbosity, saveSolution):
        #########################################
        #xlimits = np.array([[1,60], [0,10]])   #for polynomial Mooney Rivlin
        xlimits = np.array([[15,25], [0,5]])
        sampling = LHS(xlimits=xlimits)

        train_number =200
        theta = sampling(train_number)        
        ########################################
        
        self.X_init=np.zeros((train_number,2))
        Y=[]
        for i in range(0,train_number):                #initial points to GP 
                function_input=theta[i]
                self.X_init[i]=theta[i]
                Y_= np.reshape(user_function.f(data, function_input, delta,loadSteps, verbosity, saveSolution),(-1,1))
                Y.append(Y_)
            
        self.Y_init=np.reshape(np.array(Y),(-1,1))

        gpy_model = GPy.models.GPRegression(X=self.X_init, Y=self.Y_init,kernel=GPy.kern.Matern52(input_dim=self.X_init.shape[1], ARD=False))  
       
        gpy_model.optimize(messages=True)
        gpy_model.optimize_restarts(num_restarts = 1)

        return gpy_model
    
  
    def trainAll(self, delta, loadSteps, verbosity, saveSolution):
        
        self.GPs=[]
        
        for j in range(0,self.J):
            GP=self.singleGP(self.data[j],delta,loadSteps,verbosity,saveSolution)
            self.GPs.append(GP)
        return self.GPs  

#################################################################################################################
    def theta_given_phi(self, phi, theta):
        return norm.logpdf(theta[0], loc=phi[0], scale=phi[1]) + norm.logpdf(theta[1], loc=phi[2], scale=phi[3])

    def logLike(self,data, phi, delta,loadSteps, verbosity, saveSolution):
        ll = 0.0
        for j in range(self.J): # For each experiment
            tmp = 0.0
            for i in range(self.N_MC):
                theta =np.array( [np.random.normal(phi[0],phi[1]),np.random.normal(phi[2],phi[3])]) # na balw abs(theta)
                gp_theta=np.reshape(theta,(1,-1))
                part1 =self.GPs[j].predict(gp_theta) 
                part2  = self.theta_given_phi(phi, theta)
                tmp+=float(part1[0])  *part2                        # den douleuei me part2
                
            if tmp<=0:
                tmp=0.01
            ll += np.log(tmp/self.N_MC)        
        return ll             


        
#################################################################################################################

# Imagine this is your data!
data = [-0.6538075772610137,
        -1.3600898575757168,
        -2.12445260813641,
        -2.9531997892903594,
        -3.853423265511568,
        -4.833097824627032,
        -5.901174910748575,
        -7.067660481777995]

N=3 #number of experiments
#def genData(real_data):
#    data = np.array(real_data) + rnd.normalvariate(0,0.005)
    #stress=stress.tolist()
#    return data
 
    
    
Data_i=[]
#for l in range(N): # Generate Synthetic data for each experiment
#    Data_i.append(genData(data))
data_realization = Gen_data3.Generation_data()

for l in range(N): # Generate Synthetic data for each experiment
    #theta_real =np.array( [np.random.normal(phi_real[0],phi_real[1]),np.random.normal(phi_real[2],phi_real[3])])  
    #Data_i.append(data_realization.apply(theta_real,1,5,True,True))
    Data_i.append(data_realization.apply())




loadSteps = 10
delta = 0.02
saveSolution = True
verbosity = True

myModel = HyperElastic(Data_i)



# Test
#phi[0]=[0.9,0.01,0.1,0.1]
#theta =np.array( [np.random.normal(phi[0][0],phi[0][1]),np.random.normal(phi[0][2],phi[0][3])])
#kappa_keepo=myModel.apply(data,theta, delta, loadSteps, verbosity, saveSolution)



user_function, integral_bounds = myModel.press_fem()
######################################################
#check fig
#m=myModel.singleGP(Data_i[0],delta,loadSteps, verbosity, saveSolution)
#from IPython.display import display
#display(m)
#fig = m.plot()
#GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
#####################################################


tr=myModel.trainAll(delta,loadSteps,verbosity,saveSolution)
'''
######################################################################################################
ndraws = 3000  # number of draws from the distribution
phi=np.zeros((ndraws,4))
#theta=np.zeros((1,2))
prop_phi=np.zeros((ndraws,4))
sigma1_sq=np.zeros((ndraws,1))
sigma2_sq=np.zeros((ndraws,1))
a=[1]
sigma1_sq[0]=1
sigma1_sq_it=1
k=0
ni=0
m=1.1
a_star=0.60
acceptanceCount=0
phi[0]=[20,1,1,0.1]

#likelihood,theta[0]=myModel.apply(phi[0], 1, 5, True, True)
#logLikelihood=np.log(likelihood)
#theta =np.array( [np.random.normal(phi[0][0],phi[0][1]),np.random.normal(phi[0][2],phi[0][3])])
#theta=abs(theta)
#logLikelihood = myModel.log_likelihood(Data_i,phi[0],1,5,True,True)
logLikelihood = myModel.logLike(Data_i,phi[0],delta,loadSteps, verbosity, saveSolution)

for it in range(1,ndraws):
    prop_phi[it][0]=np.sqrt(1-sigma1_sq_it**2)*phi[it-1][0] + 5*sigma1_sq_it*float(truncnorm.rvs(-1,1,size=1))    
    prop_phi[it][1]=np.sqrt(1-sigma1_sq_it**2)*phi[it-1][1] + sigma1_sq_it*float(truncnorm.rvs(-1,1,size=1))
    prop_phi[it][2]=np.sqrt(1-sigma1_sq_it**2)*phi[it-1][2] + 0.3*sigma1_sq_it*float(truncnorm.rvs(-1,1,size=1))
    prop_phi[it][3]=np.sqrt(1-sigma1_sq_it**2)*phi[it-1][3] + 0.03*sigma1_sq_it*float(truncnorm.rvs(-1,1,size=1))
    
    
    prop_phi[it]=abs(prop_phi[it])
    
    #############################    
#update the adaptation
    if it>100*(ni+1):
        k=k+1
        ni=ni+1
        #update of sigma_sq
        log_sigma1_sq=np.log(sigma1_sq[k-1]) + m**(-k)*(np.mean(np.array([a]))-a_star)
        sigma1_sq[k]=np.exp(log_sigma1_sq)
        #if sigma1_sq[k]>1:
        #    sigma1_sq[k]=0.99
        sigma1_sq_it=float(sigma1_sq[k]) 

    ##############################

    #prop_likelihood,prop_theta=myModel.apply(prop_phi[it], 1, 5, True, True)   
    #prop_logLikelihood=np.log(prop_likelihood)
    
    #prop_logLikelihood = myModel.log_likelihood(Data_i,prop_phi[it],1,5,True,True)
    prop_logLikelihood = myModel.logLike(Data_i,prop_phi[it],delta,loadSteps, verbosity, saveSolution)


    alpha= prop_logLikelihood -logLikelihood
    alpha = min(0,alpha)
            
    
    if     np.log(rnd.random())<alpha:
    
           acceptanceCount=acceptanceCount+1
           phi[it]=prop_phi[it]
           #theta[it]=prop_theta
           #Likelihood=prop_Likelihood
           logLikelihood=prop_logLikelihood
           a=a+[1]
    else:
           phi[it]=phi[it-1]
           a=a+[0]
           #theta[it] = theta[it-1]


'''














                    
        
        