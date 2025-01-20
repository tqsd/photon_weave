"""
This script executes all lossy circuits in thir dir and compiles a graph
Each process consumes a lot of memory at least 32 gb is needed to run
each script with initial state 2
"""
import subprocess
import json
import time
import platform
import psutil
from cpuinfo import get_cpu_info
import matplotlib.pyplot as plt

SCRIPTS = {
    "photon_weave":"lc_photon_weave.py",
    "qutip":"lc_qutip.py",
    "qiskit": "lc_qiskit.py"
    }

system_info = {
    "OS": platform.system(),
    "OS Version": platform.version(),
    "Processor": get_cpu_info()["brand_raw"],
    "Machine": platform.machine(),
    "Python Version": platform.python_version(),
}

# Gather RAM and swap information
memory_info = psutil.virtual_memory()
swap_info = psutil.swap_memory()

# Add RAM and swap size to the system info
system_info["Total RAM"] = f"{memory_info.total / (1024 ** 3):.2f} GB"
system_info["Available RAM"] = f"{memory_info.available / (1024 ** 3):.2f} GB"
system_info["Total Swap"] = f"{swap_info.total / (1024 ** 3):.2f} GB"
system_info["Used Swap"] = f"{swap_info.used / (1024 ** 3):.2f} GB"

# Format system information for display
system_info_text = "\n".join([f"{key}: {value}" for key, value in system_info.items()])


def run_benchmark(initial_state):
    results = {}
    for name, script in SCRIPTS.items():
        try:
            print(f"Running {name}:{script}")
            start_time = time.time()
            process = subprocess.run(
                ["python", script, str(initial_state), "--lossy"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
                )
            end_time = time.time()
            execution_time = end_time - start_time

            output = json.loads(process.stdout)
            results[name] = {
                "time": execution_time,
                "output": output
                }
        except subprocess.CalledProcessError as e:
            # Handle script execution errors
            print(f"Error running {name}: {e}")
            results[name] = {
                "time": None,
                "output": None,
                "error": e.stderr
            }
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            print(f"Error parsing JSON from {name}: {e}")
            results[name] = {
                "time": None,
                "output": None,
                "error": "Invalid JSON output"
            }
    print(results)
    create_plot(results, None)


def add_system_info(fig, system_info_text):
    """
    Adds system information as a separate plot below the main plots.

    Args:
        fig (matplotlib.figure.Figure): The figure to modify.
        system_info_text (str): The system information to display.
    """
    # Create a new subplot for system info
    ax = fig.add_subplot(3, 1, 3)
    ax.axis("off")  # Turn off the axis
    ax.text(
        0.5, 0.5, system_info_text, fontsize=12, ha="center", va="center",
        bbox=dict(boxstyle="round", alpha=0.3)
    )

def create_plot(data, file_location):
    linestyles = ("solid", "solid", "dotted")
    linecolors = ("black", "red", "blue")
    steps = list(range(1, 7))  # x-axis (steps)
    labels = list(data.keys())  # Method names
    state_sizes = [data[label]['output']['state_sizes'] for label in labels]
    operator_sizes = [data[label]['output']['operator_sizes'] for label in labels]
    execution_times = [data[label]['time'] for label in labels]

    # Set up the figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), constrained_layout=True)

    # Left Plot: State Sizes (Logarithmic Scale)
    for i, label in enumerate(labels):
        ax1.plot(steps, state_sizes[i], marker='o',
                 label=f"{label} ({execution_times[i]:.2f}s)",
                 linestyle=linestyles[i], color=linecolors[i])
    ax1.set_title('State Sizes Over Steps (Log Scale)', fontsize=14)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('State Sizes (log scale)', fontsize=12)
    ax1.set_yscale('log')

    # Right Plot: Operator Sizes (Logarithmic Scale)
    for i, label in enumerate(labels):
        ax2.plot(steps, operator_sizes[i], marker='o',
                 label=f"{label} ({execution_times[i]:.2f}s)",
                 linestyle=linestyles[i], color=linecolors[i])
    ax2.set_title('Operator Sizes Over Steps (Log Scale)', fontsize=14)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Operator Sizes (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Create a single legend on the right-hand side

    # Save the figure as a PNG with 600 DPI
    plt.savefig("lossy_circuit_paper.png", dpi=600)

def acreate_plot(data, file_location):
    linestyles = ("solid", "solid", "dotted")
    linecolors = ("black", "red", "blue")
    steps = list(range(1, 7))  # x-axis (steps)
    labels = list(data.keys())  # Method names
    state_sizes = [data[label]['output']['state_sizes'] for label in labels]
    operator_sizes = [data[label]['output']['operator_sizes'] for label in labels]
    execution_times = [data[label]['time'] for label in labels]
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)

    # Top Plot: State Sizes (Logarithmic Scale)
    ax1 = fig.add_subplot(2, 1, 1)
    for i, label in enumerate(labels):
        ax1.plot(steps, state_sizes[i], marker='o',
                 label=f"{label} ({execution_times[i]:.2f}s)",
                 linestyle=linestyles[i], color=linecolors[i])
    ax1.set_title('State Sizes Over Steps (Log Scale)', fontsize=14)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('State Sizes (log scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Bottom Plot: Operator Sizes (Logarithmic Scale)
    ax2 = fig.add_subplot(2, 1, 2)
    for i, label in enumerate(labels):
        ax2.plot(steps, operator_sizes[i], marker='o',
                 label=f"{label} ({execution_times[i]:.2f}s)",
                 linestyle=linestyles[i], color=linecolors[i])
    ax2.set_title('Operator Sizes Over Steps (Log Scale)', fontsize=14)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Operator Sizes (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Dedicated Subplot for System Information
    # ax3 = fig.add_subplot(3, 1, 3)
    # ax3.axis("off")  # No axes for the system info
    # ax3.text(
    #     0.5, 0.5, system_info_text, fontsize=12, ha="center", va="center",
    #     bbox=dict(boxstyle="round", alpha=0.3)
    # )

    plt.savefig("lossy_circuit.png", dpi=600)


run_benchmark(2)
