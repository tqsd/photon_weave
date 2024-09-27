from typing import List

import jax.numpy as jnp

from photon_weave.operation import CompositeOperationType, FockOperationType, Operation
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope


def time_bin_encoding(alpha: float, beta: float) -> List[List[int]]:
    """
    Simulates time bin encoding

    Time Bin encoding makes use of four beam splitters
    and two phase shifters (see diagram below):
        ┍━[PS:a]━┑    ┍━[PS:b]━┑
        │        │    │        │
    ———[/]━━━━━━[\]━━[/]━━━━━━[\]━━◽
                               ┃
                              ◽

    Parameters
    ----------
    alpha: float
        Phase shift for the first arm
    beta: float
        Phase shift for the second arm

    Returns:
    List[List[int]]
        Measurement outcomes
        Three lists correspond to three measurement times
        Each list has two elements first corresponding to the
        measurement outcome at top detector and second corresponding
        to the bottom detector
    """
    # Create an envelope with one photon
    env1 = Envelope()
    env1.fock.state = 3

    # Create an empty envelope
    env2 = Envelope()

    bs = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
    ps1 = Operation(FockOperationType.PhaseShift, phi=alpha)
    ps2 = Operation(FockOperationType.PhaseShift, phi=beta)

    # Combine the envelopes
    ce = CompositeEnvelope(env1, env2)

    #######################
    # First Beam Splitter #
    #######################

    # Apply first beam splitter
    ce.apply_operation(bs, env1.fock, env2.fock)

    # One envelope goes through phase shift
    ce.apply_operation(ps1, env1.fock)

    ########################
    # Second Beam Splitter #
    ########################
    """
    Now the two split envelopes enter the second
    beam splitter. Because env1 travels larger
    distance, they will enter the beam splitter
    separately. In practice we feed an envelope
    with a vacuum state as the other input
    """

    # Other envelope goes through beam splitter with vacuum state
    tmp_env_0_1 = Envelope()
    tmp_env_0_2 = Envelope()

    # Add the the new vacuum states to the product space
    ce = CompositeEnvelope(ce, tmp_env_0_1, tmp_env_0_2)
    ce.combine(env2.fock, tmp_env_0_1.fock)
    ce.combine(env1.fock, tmp_env_0_2.fock)

    # Apply beam split operation for the first arriving envelope
    # (The one that traveled shorter distance)
    ce.apply_operation(bs, env2.fock, tmp_env_0_1.fock)

    # Apply beam split operation for the second arriving envelope
    ce.apply_operation(bs, tmp_env_0_2.fock, env1.fock)

    # envelopes tmp_env_0_1 and tmp_env_0_2 are dropped

    #######################
    # Third Beam Splitter #
    #######################

    """
    As in previous case now two envelopes go through the
    beamsplitter, separately
    """

    env_1_1 = Envelope()
    env_1_2 = Envelope()

    ce = CompositeEnvelope(ce, env_1_1, env_1_2)
    ce.combine(env1.fock, env_1_1.fock)
    ce.combine(env2.fock, env_1_2.fock)

    ce.apply_operation(bs, env1.fock, env_1_1.fock)
    ce.apply_operation(bs, env2.fock, env_1_2.fock)

    """ 
    In this case the envelopes do not get dropped and
    they will be measured in the end
    """

    # Apply the base shift to the top arm
    ce.apply_operation(ps2, env1.fock)
    ce.apply_operation(ps2, env2.fock)

    ########################
    # Fourth Beam Splitter #
    ########################

    """
    Now we need to correctly combine
    the envelopes in the last beamsplitter

    We have:
    - at t_1 an envelope that traveled the shortest path is entering
      the beamsplitter alone (env_1_2)
    - at t_2 an envelope which travelled trhough the short path after
      first beam splitter and longer after second one enters the beamsplitter
      together with the envelope which travelled through the longer path after
      the first beamsplitter and shorter path after the second beamsplitter.
      (env2, env_1_1)
    - at t_3 an envelope which travelled through the longer paths enters the
      beamsplitter alone (env1)
    """

    env_2_1 = Envelope()
    env_2_2 = Envelope()
    ce = CompositeEnvelope(ce, env_2_1, env_2_2)
    ce.combine(env2.fock, env_2_1.fock)
    ce.combine(env2.fock, env_1_1.fock)
    ce.combine(env1.fock, env_2_2.fock)
    # t_1
    ce.apply_operation(bs, env_1_2.fock, env_2_1.fock)
    # t_2
    ce.apply_operation(bs, env2.fock, env_1_1.fock)
    # t_3
    ce.apply_operation(bs, env1.fock, env_2_2.fock)

    #############
    # MEASURING #
    #############

    """
    Now we will measure at three different times
    """

    # t_0 At the top detector
    m_t_0_0 = env_1_2.measure()[env_1_2.fock]
    # t_0 At the bottom detector
    m_t_0_1 = env_2_1.measure()[env_2_1.fock]

    m_t0 = (m_t_0_0, m_t_0_1)

    # t_1 At the top detector
    m_t_1_0 = env2.measure()[env2.fock]
    # t_1 At the bottom detector
    m_t_1_1 = env_1_1.measure()[env_1_1.fock]

    m_t1 = (m_t_1_0, m_t_1_1)

    # t_2 At the top detector
    m_t_2_0 = env1.measure()[env1.fock]
    # t_2 At the bottom detector
    m_t_2_1 = env_2_2.measure()[env_2_2.fock]
    m_t2 = (m_t_2_0, m_t_2_1)

    return (m_t0, m_t1, m_t2)


if __name__ == "__main__":
    print(time_bin_encoding(0, 0))
