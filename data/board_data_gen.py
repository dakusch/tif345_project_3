import numpy as np
import board

n_samples = 10000
board_data = np.zeros((n_samples, 32))
for i in range(n_samples):
    print(i)
    board_data[i] = board.experiment()
np.save('board_data_', board_data)
