import time
from memory_profiler import profile


@profile
def main():
    import jax
    from photon_weave.state.custom_state import CustomState
    from photon_weave.state.composite_envelope import CompositeEnvelope

    STATE_SIZE = 5

    def run_combine():
        C = CustomState(STATE_SIZE)
        C.expand()
        C.expand()
        ce = CompositeEnvelope(C)
        accumulated_time = 0
        for i in range(4):
            c = CustomState(STATE_SIZE)
            ce = CompositeEnvelope(ce, c)

            start_time = time.time()
            ce.combine(C, c)
            end_time = time.time()
            accumulated_time += (end_time-start_time)
        print(accumulated_time)

        jax.block_until_ready(ce.product_states[0].state)

    def measure_average_time(func, loops=10):
        times = []
        for _ in range(loops):
            start_time = time.time()  # Start the timer
            func()                    # Run the process
            end_time = time.time()    # End the timer
            times.append(end_time - start_time)  # Record the elapsed time

        average_time = sum(times) / len(times)  # Compute the average time
        print(f"Average time over {loops} runs: {average_time:.5f} seconds")

    measure_average_time(run_combine)


main()
