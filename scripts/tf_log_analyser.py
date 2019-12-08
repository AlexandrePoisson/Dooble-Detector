import re
import matplotlib.pyplot as plt
import math


f = open('log.txt')
loss_pattern = "INFO:tensorflow:loss = "
map_pattern = "DetectionBoxes_Precision/mAP = "
loss = []
map_error = []

for line in f:
    if re.search(loss_pattern, line):
        loss.append(float(re.split(r'[=,]',line)[1]))
        #print(re.split(r'[=,]',line)[1])
    if re.search(map_pattern, line):
        map_error.append(float(re.split(r'[=,]',line)[1]))
        #print(re.split(r'[=,]',line)[1])


print("Min Loss: {}".format(min(loss)))
plt.subplot(211)
plt.plot(loss)
plt.title('loss')
plt.subplot(212)
plt.plot(map_error)
plt.title('map_error')
plt.show()