import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imdecode(np.fromfile(r'E:\py\钢板计数\微信图片_20220616093154.jpg', dtype=np.uint8), 0)
# cv2.imshow("img", img)
cv2.imwrite("img_gray.jpg",img)

# 均值滤波
# img_blur = cv2.medianBlur(img, 3)
# cv2.imshow("blur", img_blur)
# cv2.waitKey()


# 锐化
sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpen_image = cv2.filter2D(img, cv2.CV_32F, sharpen_op)
sharpen_image = cv2.convertScaleAbs(sharpen_image)
# cv2.imshow("sharpen_image", sharpen_image)
cv2.imwrite("sharpen.jpg",sharpen_image)

# 计算垂直梯度
dst = cv2.Sobel(sharpen_image, -1, 0, 1)
# cv2.imshow("dst", dst)
cv2.imwrite("dst.jpg", dst)

# 绘制分布图
list_ = []
for i in range(dst.shape[0]):
    list_.append(int(np.floor(sum(dst[i, :200] / 200))))
# 归一化灰度值
max_ = max(list_)
for i in range(len(list_)):
    list_[i] = list_[i] / max_

plt.plot(list_)
plt.savefig("hist.png")
plt.show()

# 根据灰度峰值统计个数
sum_ = 0
peakWidth = len(list_)/200  # 峰的宽度，用于排除临近的次峰值
id = 0  # 标记最近的峰位置
for i in range(1,len(list_)):
    if list_[i] > 0.4:
        if i-id > peakWidth:
            sum_ += 1
            id = i
print("sum = ", sum_)

# 根据阈值计算个数
# total = 0
# mark = 0  # 用来判断是否第一次遇到阈值
# for i in range(dst.shape[0]):
#     if dst[i, 0] > 200:
#         if mark == 0:
#             total += 1
#             mark = 1
#         else:
#             continue
#     else:
#         mark = 0
#
# print("total = ", total)

# # 二值化
# binary = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 10)
# # binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("binary", binary)


cv2.waitKey()
cv2.destroyAllWindows()
