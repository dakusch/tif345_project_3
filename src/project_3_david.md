```python
import numpy as np
import matplotlib.pyplot as plt
from galton_utils import create_combined_plot, simulate_multiple_beads
%matplotlib inline
```

# Galton Board Simulation Utilities Documentation

This module contains utilities for simulating and visualizing a Galton Board with memory effect on a rocking ship.

## Simulation Functions

### `simulate_single_bead(alpha, s, n_rows=31)`
Simulates the path of a single bead through the Galton board.
- **Parameters:**
  - `alpha`: Memory effect parameter [0, 0.5]
  - `s`: Ship tilt parameter [-0.25, 0.25]
  - `n_rows`: Number of rows in the board (default: 31)
- **Returns:** Final position of the bead (0 to n_rows)

### `simulate_multiple_beads(n_beads, alpha, s, n_rows=31)`
Simulates multiple beads through the Galton board.
- **Parameters:**
  - `n_beads`: Number of beads to simulate
  - `alpha`: Memory effect parameter [0, 0.5]
  - `s`: Ship tilt parameter [-0.25, 0.25]
  - `n_rows`: Number of rows in the board (default: 31)
- **Returns:** Array with the count of beads in each final position

## Visualization Functions

### `plot_galton_board(ax)`
Creates a stylized visualization of the Galton board structure.
- **Parameters:**
  - `ax`: Matplotlib axes object to plot on

### `plot_bead_distribution(positions, counts, ax)`
Plots the distribution of beads as a bar chart.
- **Parameters:**
  - `positions`: Array of x-positions (typically 0 to 31)
  - `counts`: Array of bead counts at each position
  - `ax`: Matplotlib axes object to plot on

### `create_combined_plot(counts, title=None)`
Creates a combined visualization with both the Galton board and bead distribution.
- **Parameters:**
  - `counts`: Array of bead counts at each position
  - `title`: Optional title for the plot
- **Returns:** Tuple of (figure, (ax1, ax2))

### `plot_distribution_only(counts, title=None)`
Creates a single plot showing just the bead distribution.
- **Parameters:**
  - `counts`: Array of bead counts at each position
  - `title`: Optional title for the plot
- **Returns:** Tuple of (figure, ax)

## Usage Example:
```python
# Run simulation
results = simulate_multiple_beads(n_beads=1000, alpha=0.2, s=0.1)

# Create visualization
fig, axes = create_combined_plot(
    results, 
    'Galton Board Simulation'
)
plt.show()


```python
data = np.load('board_data_.npy')

#plot mean of rows in bar chart
mean_data = np.mean(data, axis=0)
plt.bar(np.arange(len(mean_data)), mean_data, color = 'blue', alpha = 0.7)
plt.xlabel('Position')
plt.ylabel('Mean number of beads')
plt.title('Galton Board mean result from data\n taken on a rocking ship with 1000 beads')
plt.show()
```


    
![png](project_3_david_files/project_3_david_2_0.png)
    



```python
from galton_utils import plot_distribution_only

# plot single experiment
counts = data[0]

plot_distribution_only(counts, 'First experiment')



```




    (<Figure size 1200x400 with 1 Axes>,
     <Axes: title={'center': 'First experiment'}, xlabel='Position', ylabel='Number of beads'>)




    
    



```python
# Test parameters
n_beads = 1000
alpha_test = 0.2
s_test = 0.1

# Run simulation
results = simulate_multiple_beads(n_beads, alpha_test, s_test)

# Create combined plot
fig, axes = create_combined_plot(
    results, 
    f'Galton Board Simulation\n({n_beads} beads, α={alpha_test}, s={s_test})'
)
plt.show()

# Print some statistics
print(f"Total beads: {np.sum(results)}")
print(f"Mean position: {np.average(range(len(results)), weights=results):.2f}")
```


    
    


    Total beads: 1000
    Mean position: 19.28



```python
import numpy as np
from galton_utils import simulate_multiple_beads

def calculate_summary_statistics(distribution):
    """
    Calculate summary statistics for a bead distribution
    Returns: (mean position, standard deviation)
    """
    positions = np.arange(len(distribution))
    mean = np.average(positions, weights=distribution)
    # Calculate weighted standard deviation
    variance = np.average((positions - mean)**2, weights=distribution)
    std = np.sqrt(variance)
    return mean, std
```

    
    No memory, No tilt:
    Means: 15.51 ± 0.06
    Stds: 2.76 ± 0.05
    
    Strong memory, No tilt:
    Means: 15.52 ± 0.20
    Stds: 4.21 ± 0.07
    
    No memory, Right tilt:
    Means: 21.68 ± 0.05
    Stds: 2.55 ± 0.01
    
    Strong memory, Right tilt:
    Means: 31.00 ± 0.00
    Stds: 0.00 ± 0.00


### Solve task 2 by using abc rejection algo


```python
import numpy as np
import matplotlib.pyplot as plt
from galton_utils import simulate_multiple_beads
from multiprocessing import Pool
from tqdm import tqdm

def calculate_summary_statistics(distribution):
    """Calculate summary statistics for a single bead distribution."""
    positions = np.arange(len(distribution))
    mean = np.average(positions, weights=distribution)
    variance = np.average((positions - mean)**2, weights=distribution)
    std = np.sqrt(variance)
    return (mean, std)  # Explicitly return as tuple

def gaussian_kernel(x, h):
    """Compute Gaussian kernel."""
    return np.exp(-0.5 * (x/h)**2)

def compute_distance(stats1, stats2):
    """Compute distances between two sets of summary statistics."""
    mean1, std1 = stats1
    mean2, std2 = stats2
    mean_dist = np.abs(mean1 - mean2)
    std_dist = np.abs(std1 - std2)
    return mean_dist, std_dist

def simulate_and_evaluate(args):
    """Function to run one simulation and evaluation.
    
    Args:
        args: tuple containing (observed_stats, h)
    """
    observed_stats, h = args
    
    # Generate proposals
    alpha_proposal = np.random.uniform(0, 0.5)
    s_proposal = np.random.uniform(-0.25, 0.25)
    
    # Simulate
    simulated_data = simulate_multiple_beads(1000, alpha_proposal, s_proposal)
    simulated_stats = calculate_summary_statistics(simulated_data)
    
    # Compare
    mean_dist, std_dist = compute_distance(observed_stats, simulated_stats)
    kernel_value = gaussian_kernel(mean_dist, h) * gaussian_kernel(std_dist, h)
    
    # Return None if rejected, alpha if accepted
    if np.random.random() < kernel_value:
        return alpha_proposal
    return None

def abc_rejection_parallel(observed_data, n_iterations=1000, h=0.1, n_cores=None):
    """Parallel ABC rejection algorithm."""
    # Calculate observed summary statistics
    observed_stats = calculate_summary_statistics(observed_data)
    
    # Create list of arguments for parallel processing
    args_list = [(observed_stats, h) for _ in range(n_iterations)]
    
    # Run parallel simulations with progress bar
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(simulate_and_evaluate, args_list),
            total=n_iterations,
            desc="Running ABC"
        ))
    
    # Filter out None values (rejected samples)
    accepted_alpha = [x for x in results if x is not None]
    
    return accepted_alpha

# Load and prepare data
data = np.load('board_data_.npy')
mean_data = np.mean(data, axis=0)

# Run parallel ABC algorithm
alphas = abc_rejection_parallel(mean_data, n_iterations=100000, h=4)

# Visualize results
plt.figure(figsize=(10, 6))
plt.hist(alphas, bins=30, density=True)
plt.xlabel('α (Memory Effect)')
plt.ylabel('Density')
plt.title('Posterior Distribution of α')
plt.show()

# Print statistics
print(f"Estimated α (mean): {np.mean(alphas):.3f}")
print(f"95% Credible Interval: [{np.percentile(alphas, 2.5):.3f}, {np.percentile(alphas, 97.5):.3f}]")
print(f"Number of accepted samples: {len(alphas)}")
print(f"Acceptance rate: {len(alphas)/1000:.1%}")
```

    Running ABC: 100%|██████████| 100000/100000 [09:16<00:00, 179.56it/s]
    


    Estimated α (mean): 0.246
    95% Credible Interval: [0.013, 0.488]
    Number of accepted samples: 27983
    Acceptance rate: 2798.3%

