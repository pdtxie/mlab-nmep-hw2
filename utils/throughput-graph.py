import matplotlib.pyplot as plt

throughputs = [0.0013812180455541737,
               0.002562828360466591,
               0.005341089326729775,
               0.009305563589114227,
               0.00977617082431183,
               0.010232591904909273,
               0.010149745961124635,
               0.00987455731541048,
               0.009809642008141143,
               0.008485357314357575,
               0.00853326870419319,
               0.008169735449740947,
               0.004633595279515382,
               0.002583539860469085,
               0.0013650913027638508]

batch_sizes = list(map(str, [2**i for i in range(15)]))

plt.figure(figsize=(10, 6))
plt.bar(x=batch_sizes, height=throughputs, width=0.3)
plt.xlabel("batch size")
plt.ylabel("throughput")

plt.savefig("img.png")
