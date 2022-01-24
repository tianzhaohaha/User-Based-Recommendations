import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["font.sans-serif"] = ['SimHei']

x1 = [500,1000,2000,4000,8000]
y1 = [98.69,94.08,92.52,84.17,77.29]
x2 = [2,5,10,15,20]
y2 = [68.94,67.51,66.44,65.29,65.79]

plt.plot(x2,y2,'g')
plt.title("dim&SSE", fontproperties="SimHei")
plt.xlabel('fim', fontproperties="SimHei")
plt.ylabel('SSE')
plt.legend()
plt.show()
