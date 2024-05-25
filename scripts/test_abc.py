import afilter
import matplotlib.pyplot as plt
import sigutil as util

fs, d, d_norm = util.read_normalized('data/2_RecStatic.wav')  # read d[n]
_, x, x_norm = util.read_normalized('data/2_Sig.wav')  # read x[n]

# Apply the iteration (cf other notebooks)
N_bees = 10
limit = 15
K = 600

f_ad, reconstructed = afilter.adaptive_abc(x_norm, d_norm, K, N_bees, limit)

# Create the plot
plt.figure(figsize=(12, 6))

# Plotting with thicker lines and markers
plt.plot(reconstructed, label='e[n]', linestyle=':', linewidth=2, color='r', marker='x', markersize=3, markevery=50)
plt.plot(d_norm + 0.5, label='d[n]', linewidth=2, linestyle='-', marker='o', markersize=3, markevery=50)

areas = [(5000, 20000), (35000, 50000), (75000, 115000), (155000, 187000), (210000, 240000)]

for a, b in areas:
    plt.axvspan(a, b, color='gray', alpha=0.3)
# Add labels and title
plt.xlabel('Sample')
plt.ylabel('Amplitude (blue shifted)')
plt.title('Adaptive, ABC approach')
plt.legend()  # Add a legend to distinguish the plots

# Show grid
plt.grid(True)

# Show the plot
plt.show()