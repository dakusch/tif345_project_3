# galton_utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_galton_board(ax):
    """Creates a stylized Galton board visualization."""
    rows = 31
    for row in range(rows):
        y = 1 - (row / rows)
        pins_in_row = row + 1
        for pin in range(pins_in_row):
            x = (pin - (pins_in_row-1)/2) / 16 + 0.5
            ax.plot(x, y, 'o', color='blue', markersize=5)
    
    ax.axis('off')
    ax.set_xlim(-0.5-1.5/16, 1.5+1.5/16)

def plot_bead_distribution(positions, counts, ax):
    """Plots the distribution of beads."""
    bars = ax.bar(positions, counts, color='blue', alpha=0.7)
    
    for i in range(len(positions) + 1):
        ax.axvline(x=i-0.5, color='black', linestyle='-', linewidth=5)
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Number of beads')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_combined_plot(counts, title=None):
    """Creates a combined plot with Galton board and bead distribution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                  gridspec_kw={'height_ratios': [1, 1]})
    
    plot_galton_board(ax1)
    
    positions = range(len(counts))
    plot_bead_distribution(positions, counts, ax2)
    
    plt.subplots_adjust(hspace=0, top=0.85)
    
    if title:
        plt.suptitle(title, y=0.98, fontsize=14)
    
    return fig, (ax1, ax2)

def plot_distribution_only(counts, title=None):
    """Creates a single plot of just the bead distribution."""
    fig, ax = plt.subplots(figsize=(12, 4))
    positions = range(len(counts))
    plot_bead_distribution(positions, counts, ax)
    
    if title:
        plt.title(title)
    
    return fig, ax

def simulate_single_bead(alpha, s, n_rows=31):
    """Simulates the path of a single bead."""
    current_pos = 0  
    previous_M = 0  
    
    for row in range(n_rows):
        P_right = 0.5 + (alpha * previous_M + s)
        goes_right = np.random.random() < P_right
        
        if goes_right:
            current_pos += 1
            previous_M = 0.5  
        else:
            previous_M = -0.5 
            
    return current_pos

def simulate_multiple_beads(n_beads, alpha, s, n_rows=31):
    """Simulates multiple beads and returns their final positions."""
    positions = [simulate_single_bead(alpha, s, n_rows) for _ in range(n_beads)]
    counts = np.bincount(positions, minlength=n_rows+1)
    return counts