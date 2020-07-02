import matplotlib.pyplot as plt
from dolfin import *
import random as rnd
from RandomProcess import Matern52

import numpy as np
from scipy.stats import truncnorm
from numpy import where
###############################################################################
#from smt.sampling_methods import LHS


import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import math

import random as rnd
import seaborn as sns
from typing import Tuple, List, Optional
from scipy.stats import norm

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


class Generation_data():

    def __init__(self, mean = 0.0, stdev = 1.0,ell = 0.1, mkl = 32, N = 20):

        #

        self.mkl = mkl

        self.ell = ell

        self.mean = mean
        self.stdev = stdev

        self.nu = 0.3

        # Create mesh and define function space
        self.mesh = UnitCubeMesh(N, N, N)

        self.V = VectorFunctionSpace(self.mesh, "Lagrange", 1)

        self.loadSteps = 10   # cube [1,1,1] is pressed by 0.2 on top
        self.delta = 0.02

        self.verbosity = True

        # Find the top and bottom of my domain
        self.top =  CompiledSubDomain("near(x[2], side) && on_boundary", side = 1.0)
        self.bottom = CompiledSubDomain("near(x[2], side) && on_boundary", side = 0.0)
        self.c = Constant((0.0, 0.0, 0.0))
        self.bc_bottom = DirichletBC(self.V, self.c, self.bottom) #boundary for bottom face = fixed

        # Find dofs on top face
        #ff = MeshFunction("size_t",mesh, mesh.topology().dim()-1, 0)
        #top.mark(ff, 1)
        self.vv = Function(self.V)
        self.bc = DirichletBC(self.V, Constant((0.0, 0.0, 1.0)), self.top)
        self.bc.apply(self.vv.vector())
        self.top_dofs = where(self.vv.vector()==1.0)[0]

    def computeEigenVectors(self):

        # Set up the random process on a SCALAR functionspace.
        self.VV = FunctionSpace(self.mesh, "Lagrange", 1)
        _dof_coords = self.VV.tabulate_dof_coordinates().reshape((-1, 3))
        _dof_indices = self.VV.dofmap().dofs()
        self.coords = _dof_coords[_dof_indices, :]

        self.random_process = Matern52(self.coords, self.ell)
        self.random_process.compute_eigenpairs(self.mkl)

    def apply(self, parameters, plotField = False, fieldFileName = "RandomField.vtu", plotSolution = False, solutionFileName = "solution"):


        ##########################################################################################

        self.random_process.generate(self.mean, self.stdev, parameters)
        ##########################################################################################

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

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        #######################################################
        # Elasticity parameters
        E = Function(self.VV)
        E.vector()[:] = np.exp(self.random_process.random_field)
        if(plotField):
            File("elasticity_gp0.pvd") << E
        mu, lmbda = E/(2*(1 + self.nu)), E * self.nu/((1 + self.nu)*(1 - 2 * self.nu))
        ########################################################

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

        # Total potential energy
        Pi = psi * dx

        # Compute first variation of Pi (directional derivative about u in the direction of v)
        F = derivative(Pi, u, v)

        # Compute Jacobian of F
        J = derivative(F, u, du)

        # Loading loop
        Force=[]
        for i in range(1, self.loadSteps + 1):

                    if(self.verbosity):
                        print("Load step - " + str(i))

                    # Update non-homogeneous boundary condition for current load step
                    r = Constant((0.0, 0.0, -i * self.delta))
                    bc_top = DirichletBC(self.V, r, self.top)
                    bcs = [self.bc_bottom, bc_top]

                    # Solve variational problem
                    solve(F == 0, u, bcs, J=J)

                    # Save solution in VTK format
                    if(plotSolution):
                        file = File(solutionFileName + str(i) + ".pvd");
                        file << u;

                    # Output forces
                    y = assemble(F)
                    Force_top = 0
                    for i in self.top_dofs:
                        Force_top += y[i]

                    print(Force_top)
                    Force.append(Force_top)

        return Force

N=1 #number of experiments
Data_i=[]
data_realization = Generation_data(N = 5, mkl = 10)

data_realization.computeEigenVectors()

np.random.seed(123) # Fix seed to generates same experiments each time.

param = np.random.normal(size=(data_realization.mkl, N))

for i in range(N): # Generate Synthetic data for each experiment
    Data_i.append(data_realization.apply(param[:,i]), plotField = True, fieldFileName = ("RandomField" + str(i) + ".vtu"))
