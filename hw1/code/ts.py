import numpy as np

a = np.array([1,2,3])
for num in a:
    num = 10
a = np.array([[1,2,3,4],[5,6,7,8]])
b = np.array([1,2,3,4,5,6,7,8,9])
c = np.array([8]).astype(np.uint8)[0]

print(8 == c)
print(type(c))

for i in range(10/2):
    print(i)