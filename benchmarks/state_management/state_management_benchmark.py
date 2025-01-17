import subprocess
import glob
import os
import matplotlib.pyplot as plt


def run_mprof(file_list):
    """Run mprof for a list of script files."""
    data_files = []
    for script in file_list:
        output_file = f"{script}.dat"
        print(f"Running mprof for {script}...")

        # Run mprof and save the output file
        subprocess.run(
            ["mprof", "run", "--interval", "0.01", "-o", output_file, "python", script],
            check=True,
        )
        data_files.append(output_file)
        print(f"mprof output saved to {output_file}")
    return data_files


def read_mprof_data(filename):
    """Read mprof data from a .dat file and return timestamps and memory usage."""
    with open(filename, "r") as f:
        lines = f.readlines()

    # Skip header lines
    data = [line.strip().split() for line in lines if not line.startswith("#")]

    # Extract timestamps and memory usage
    timestamps = [float(d[2]) for d in data[1:]]
    memory = [float(d[1]) for d in data[1:]]

    return timestamps, memory


def plot_memory_usage(data_files):
    """Plot memory usage for all .dat files."""
    plt.figure(figsize=(10, 6))
    for data_file in data_files:
        script_name = os.path.splitext(os.path.basename(data_file))[0]
        timestamps, memory = read_mprof_data(data_file)

        # Align timestamps to start from 0
        timestamps = [t - timestamps[0] for t in timestamps]

        # Plot memory usage
        plt.plot(timestamps, memory, label=script_name)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("memory_comparison.png")
    print("Memory usage plot saved as memory_comparison.png")
    plt.savefig("state_management.png")


def remove_dat_files():
    # Find all .dat files in the current directory
    dat_files = glob.glob("*.dat")

    for dat_file in dat_files:
        print(f"Removing file: {dat_file}")
        os.remove(dat_file)  # Delete the file


if __name__ == "__main__":
    # List of Python scripts to benchmark
    scripts_to_run = [
        "state_management_photon_weave.py",
        "state_management_qutip.py",
        "state_management_qiskit.py",
    ]

    # Run mprof benchmarks
    dat_files = run_mprof(scripts_to_run)

    # Plot the results
    plot_memory_usage(dat_files)

    # Remove dat files
    remove_dat_files()
