from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import PolarizationLabel
from photon_weave.operation.operation import Operation
from photon_weave.operation.composite_operation import CompositeOperationType
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Callable, Optional
from photon_weave.state.expansion_levels import ExpansionLevel
import matplotlib.pyplot as plt
from photon_weave.state.base_state import BaseState
import jax.numpy as jnp

def  PBS(env1:Optional[Envelope] = None, env2: Optional[Envelope] = None) -> dict[str, Envelope]:
    for env in [env1, env2]:
        if env:
            assert env1.polarization.index == None, "Polarization must not be in a product state"
        else:
            env1 = Envelope()
    env3 = Envelope()
    env3.polarization.state = PolarizationLabel.V
    env4 = Envelope()
    env4.polarization.state = PolarizationLabel.V

    #Operations
    H = jnp.array([[1],[0]])
    P_HH = jnp.array([[1,0], [0,0]])  #H -> H
    P_HV = jnp.array([[0,1], [0,0]])  #V -> H
    #Computing Beam Splitter Angle Port 1
    if env1.polarization.expansion_level == ExpansionLevel.Label: 
        env1.polarization.expand()
    if env1.polarization.expansion_level == ExpansionLevel.Vector:
        H_inner_1 = abs(jnp.vdot(H, env1.polarization.state))
        env1.polarization.contract()
    elif env1.polarization.expansion_level == ExpansionLevel.Matrix:
        H_inner_1 = H.conj().T @ (env1.polarization.state @ H)
        H_inner_1 = float(jnp.sqrt(jnp.abs(H_inner_1[0][0])))
     
    env1.polarization.apply_kraus([P_HH,P_HV])
    eta_1 = jnp.arccos(H_inner_1)

    #Computing Beam Splitter Angle Port 2
    if env2.polarization.expansion_level == ExpansionLevel.Label: 
        env2.polarization.expand()
    if env2.polarization.expansion_level == ExpansionLevel.Vector:
        H_inner_2 = abs(jnp.vdot(H, env2.polarization.state))
        env2.polarization.contract()
    elif env2.polarization.expansion_level == ExpansionLevel.Matrix:
        H_inner_2 = H.conj().T @ (env2.polarization.state @ H)
        H_inner_2 = float(jnp.sqrt(jnp.abs(H_inner_2[0][0])))
    env2.polarization.apply_kraus([P_HH, P_HV])
    eta_2 = jnp.arccos(H_inner_2)

    #Creating Beam Splitting Operations
    op_1 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta = eta_1)
    op_2 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta = eta_2)

    #Applying Beam Splitting Operations
    ce = CompositeEnvelope(env1, env2, env3, env4)
    ce.apply_operation(op_1, env1.fock, env3.fock)
    ce.apply_operation(op_2, env2.fock, env4.fock)


    return {
       '3H': env1, '3V': env4,
       '4H': env2, '4V': env3
    }

def measure_vector_probabilities(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    prob_callback: Optional[Callable[[BaseState, jnp.ndarray], None]] = None
) -> Dict[BaseState, jnp.ndarray]:
    
    state_objs = list(state_objs)
    # Validate that the product state has the expected shape.
    expected_dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
    assert product_state.shape == (expected_dims, 1), (
        f"Expected shape: ({expected_dims}, 1), received {product_state.shape}"
    )
    # Reshape the product state into a tensor where each mode corresponds to a state.
    shape = [s.dimensions for s in state_objs] + [1]
    psi_tensor = product_state.reshape(shape)

    probabilities_dict = {}
    for state in target_states:
        probs = []
        dims = state.dimensions
        for outcome in range(dims):
            slicer = [slice(None)] * len(psi_tensor.shape)
            slicer[state_objs.index(state)] = outcome
            psi_slice = psi_tensor[tuple(slicer)]
            probs.append(jnp.sum(jnp.abs(psi_slice) ** 2))

        probs = jnp.array(probs)
        probs /= jnp.sum(probs)
        if prob_callback:
            prob_callback(state, probs)
        probabilities_dict[state] = probs

    return probabilities_dict


if __name__ == "__main__": 
    
    env1 = Envelope()
    env1.fock.state = 3
    env1.polarization.state = PolarizationLabel.D
    env2 = Envelope()
    env2.fock.state = 0
    out = PBS(env1, env2)
    
    out['3H'].fock.resize(4)
    out['3V'].fock.resize(4)
    out['4H'].fock.resize(4)
    out['4V'].fock.resize(4)
    
    out['3H'].fock.uid = "3H"
    out['3V'].fock.uid = "3V"
    out['4H'].fock.uid = "4H"
    out['4V'].fock.uid = "4V"
    state_objs  = [out['3H'].fock, out['3V'].fock, out['4H'].fock, out['4V'].fock]
    ce = CompositeEnvelope(*[f.envelope for f in state_objs])
    ce.combine(*state_objs)
    ce.reorder(*state_objs)
    my_states = set(state_objs)
    product_state = [ps.state for ps in ce.product_states if my_states.issubset(set(ps.state_objs))][0]
    target_states = state_objs
    outcomes= measure_vector_probabilities(state_objs,target_states,product_state)
    states = list(outcomes.keys())  # ['3H', '3V', '4H', '4V']
    prob_matrix = jnp.array([outcomes[state] for state in states])
    
    #Plot results
    photon_counts = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = ["#136387", "#136387", "#136387", "#136387"]  # Sky blue, teal, lime green, steel blue

    for i, state in enumerate(states):
        row, col = divmod(i, 2)  # Determine grid position
        ax = axes[row, col]
        
        # Bar plot for each state
        ax.bar(photon_counts, prob_matrix[i], color= colors[i] ,width=0.5)

        # Formatting
        ax.set_title(f"Probability for {state}")
        ax.set_xlabel("Number of Photons", fontsize=16)
        ax.set_ylabel("Probability",fontsize=16)
        ax.set_xticks(photon_counts)
        
        ax.tick_params(axis='x', labelsize=12)  # X-axis numbers
        ax.tick_params(axis='y', labelsize=12)  # Y-axis numbers
    plt.tight_layout()
    plt.show()
