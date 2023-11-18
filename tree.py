import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz

digits = datasets.load_digits()
X ='E:/py/hand_written/dataset/t10k-images.idx3-ubyte'
y = 'E:/py/hand_written/dataset/t10k-labels.idx1-ubyte'

# 把数据所代表的图片显示出来
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(8, 8), dpi=200)  # 设置figsize为8*8，分辨率dpi为200
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Digit: %i' % label, fontsize=20)
plt.show()  # 操作环境在pycharm，所以放在循环外，绘制在同一张图片上

# 打印图片的数量和尺寸，方便查看以及后续操作
print("图片的数量和尺寸为: {0}".format(digits.images.shape))
print("图片数据的尺寸为: {0}".format(digits.data.shape))
# 把数据分成训练数据集和测试数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用决策树模型
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, Y_train)

# 评估模型的准确度
Y_pre = model.predict(X_test)
accuracy_score(Y_test, Y_pre)
# 打印模型的精确度
print(model.score(X_test, Y_test))

# 查看预测的情况,采用4*4张数据图片来预测和label对比
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    # 图像预测正确则数字为绿色，不正确的预测则为红色
    ax.text(0.05, 0.05, str(Y_pre[i]), fontsize=32,
            transform=ax.transAxes,
            color='green' if Y_pre[i] == Y_test[i] else 'red')
    # 标签数据设置为黑色放在图像右下角
    ax.text(0.8, 0.05, str(Y_test[i]), fontsize=32,
            transform=ax.transAxes,
            color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()