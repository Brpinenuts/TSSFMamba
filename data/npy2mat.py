import numpy as np
from scipy.io import savemat
# data_list = ['boat3', 'boat7', 'airplane2',  'airplane13', 'airplane15']
data_list = ['hsi_video']
for i in data_list:
    data = np.load((i+'.npy'))
    savemat((i+'.mat'), {'map': data})

    # data = np.load((i + '_gt' + '.npy'))
    # savemat((i + '_gt' + '.mat'), {'map': data})
print("----------------------------------")
