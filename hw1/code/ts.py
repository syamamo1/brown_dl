import numpy as np
from matplotlib import pyplot as plt

a = np.array([1,2,3])
for num in a:
    num = 10
a = np.array([[1,2,3,4],[5,6,7,8]])
b = np.array([1,2,3,4,5,6,7,8,9])
c = np.array([8]).astype(np.uint8)[0]

d = np.zeros((3,2))
d[:,1] += 1

e = np.array([np.zeros(3)]*2)
print(e)

print('hiiii')

j = np.zeros(3)
k = np.zeros((1,3))
print(j, k)


fig, axs = plt.subplots(ncols=5)
plt.show(fig)
# print('here')
# fig.suptitle("PL = Predicted Label\nAL = Actual Label")
# print('here')
# for ind, ax in enumerate(axs):
#     ax.imshow(images[ind], cmap="Greys")
#     ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
#     plt.setp(ax.get_xticklabels(), visible=False)
#     plt.setp(ax.get_yticklabels(), visible=False)
#     ax.tick_params(axis='both', which='both', length=0)
# print('here')
# plt.show()
# print('here')