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

import _pickle as cPickle

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


class MacroModel():

    def __init__(self, N = 5):

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

    def setData(self, data, sigma_f):
        self.data = data
        self.sigma_f = sigma_f

    def apply(self, param, plotSolution = False, solutionFileName = "solution"):

        ## Unwrap parameters

        C01 = param[0]
        C10 = param[1]
        C11 = param[2]
        D1 = param[3]

        # Define functions
        du = TrialFunction(self.V)            # Incremental displacement
        v  = TestFunction(self.V)             # Test function
        u  = Function(self.V)                 # Displacement from previous iteration

        # Kinematics
        d = u.geometric_dimension()
        I = Identity(d)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        I1 = tr(C)
        J  = det(F)

        I2 = 0.5 * ( I1**2 - tr(C * C))

        barI1 = J**(-2./3.) * I1

        barI2 = J**(-4./3.) * I2

        # Stored strain energy density (compressible neo-Hookean model)
        psi = C10 * (barI1 - 3.)  + C01 * (barI2 - 3.) + C11 * (barI1 - 3.) * (barI2 - 3.) + D1 * (J - 1.)**2

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

        #
        assert len(Force) == len(self.data), "Force list not same length as data list"

        return -np.sum((np.asarray(Force) - np.asarray(self.data))**2)/ (2.0 * self.sigma_f**2)


compress_data = cPickle.load( open( "CompressionData.p", "rb" ) )

numExperiments = len(compress_data)
