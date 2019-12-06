import re
import matplotlib.pyplot as plt
import math


f = open('log.txt')
pattern = "INFO:tensorflow:loss = "
loss = []
for line in f:
    if re.search(pattern, line):
        loss.append(math.log(float(re.split(r'[=,]',line)[1])))
        print(re.split(r'[=,]',line)[1])
        
    
plt.plot(loss)
plt.show()