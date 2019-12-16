import re
import matplotlib.pyplot as plt
import math


f = open('log.txt')
loss_pattern = "INFO:tensorflow:loss = "
map_pattern = "DetectionBoxes_Precision/mAP = "
loss = []
log_loss = []
map_error = []

for line in f:
    if re.search(loss_pattern, line):
        loss.append(float(re.split(r'[=,]',line)[1]))
        log_loss.append(math.log(float(re.split(r'[=,]',line)[1])))
        #print(re.split(r'[=,]',line)[1])
    if re.search(map_pattern, line):
        map_error.append(float(re.split(r'[=,]',line)[1]))
        #print(re.split(r'[=,]',line)[1])


print("Min Loss: {}".format(min(loss)))
plt.subplot(311)
plt.plot(loss)
plt.title('loss')

plt.subplot(312)
plt.plot(log_loss)
plt.title('log loss')

plt.subplot(313)
plt.plot(map_error)
plt.title('map_error')
plt.show()