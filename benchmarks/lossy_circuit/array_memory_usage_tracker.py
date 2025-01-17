class ArrayMemoryUsageTracker:
    def __init__(self):
        self.operator_sizes = []
        self.state_sizes = []

    def record_state_size(self, *states):
        mem = 0
        for s in states:
            mem += s.state.nbytes
        self.state_sizes.append(mem)

    def record_operator_size(self, *operators):
        mem = 0
        for op in operators:
            mem += op.operator.nbytes
        self.operator_sizes.append(mem)
