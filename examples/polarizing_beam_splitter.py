from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import PolarizationLabel
from photon_weave.operation.operation import Operation
from photon_weave.operation.composite_operation import CompositeOperationType
from photon_weave.state.fock import Fock, FockOperationType
from photon_weave.state.polarization import Polarization  
from photon_weave.state.polarization import PolarizationOperationType
import random
import jax.numpy as jnp


#PORT 1
env1 = Envelope()
env1.fock.state = 1
env1.polarization.state = PolarizationLabel.D

print("Input")
print("Port 1: ")
print("The number of photons in",env1.polarization ,"polarization is: ", env1.fock.state)

env1.polarization.expand()

env2 = Envelope()
env2.polarization.expand()
env2.polarization.state = jnp.array(
  [jnp.conj(env1.polarization.state[1]),
   jnp.conj(env1.polarization.state[0])]
)




#Operations
H = jnp.array([1,0])
V = jnp.array([0,1])
P_HH = jnp.array([[1,0], [0,0]])  #H -> H
P_VV = jnp.array([[0,0], [0,1]])  #V -> V 
P_VH = jnp.array([[0,1], [0,0]])  #V -> H
P_HV = jnp.array([[0,0], [1,0]])  #H -> V

op_p_HH = Operation(PolarizationOperationType.Custom, operator = P_HH)
op_p_VV = Operation(PolarizationOperationType.Custom, operator = P_VV)
op_p_VH = Operation(PolarizationOperationType.Custom, operator = P_VH)
op_p_HV = Operation(PolarizationOperationType.Custom, operator = P_HV)
H_inner_1 = abs(jnp.vdot(H, env1.polarization.state))
eta_1 = jnp.arccos(H_inner_1)
op_1 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta = eta_1)
ce_1 = CompositeEnvelope(env1, env2)
ce_1.apply_operation(op_1, env1.fock, env2.fock)


if(H_inner_1 == 0): #if env1 has been initialized as V polarization 
  env1.polarization.apply_operation(op_p_VH)  #V -> H
  env2.polarization.apply_operation(op_p_HV)  #H -> V
else:
  env1.polarization.apply_operation(op_p_HH)  #Projection onto H 
  env2.polarization.apply_operation(op_p_VV)  #Projection onto V

#PORT 2
env3 = Envelope()
env3.fock.state = 1
env3.polarization.state = PolarizationLabel.R
print("Port 2")
print("The number of photons in",env3.polarization ,"polarization is: ", env3.fock.state)
env3.polarization.expand()


env4 = Envelope()
env4.polarization.expand()
env4.polarization.state = jnp.array(
  [jnp.conj(env3.polarization.state[1]),
   jnp.conj(env3.polarization.state[0])]
)


H_inner_2 = abs(jnp.vdot(H, env3.polarization.state))
eta_2 = jnp.arccos(H_inner_2)
ce_2 = CompositeEnvelope(env3, env4)
op_2 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta = eta_2)
ce_2.apply_operation(op_2, env3.fock, env4.fock)

if(H_inner_2 == 0):
  env3.polarization.apply_operation(op_p_VH)  #V -> H
  env4.polarization.apply_operation(op_p_HV)
else:
  env3.polarization.apply_operation(op_p_HH)
  env4.polarization.apply_operation(op_p_VV)


#PBS action
ce_3 = CompositeEnvelope(env1, env4) #Consists of H polarized photons from port 1, and V polarized photons from port 2
ce_4 = CompositeEnvelope(env3, env2) #Consists of H polarized photons from port 2, and V polarized photons from port 1

print("Output")
outcome_1 = ce_3.measure(env1.fock, env4.fock)
print("Port 1: ")
print("The number of photons in",env1.polarization ,"polarization is: ", outcome_1[env1.fock])
print("The number of photons in",env4.polarization, "polarization is:",  outcome_1[env4.fock])
outcome_2 = ce_4.measure(env3.fock, env2.fock)
print("Port 2")
print("The number of photons in",env3.polarization ,"polarization is: ", outcome_2[env3.fock])
print("The number of photons in",env2.polarization, "polarization is:",  outcome_2[env2.fock])
