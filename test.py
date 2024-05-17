import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# 加载数据
csv_file = "E:\\数据管理第六次实验（用完可以删）\\第6次实验文件\\heart.csv"
data = pd.read_csv(csv_file)

# 定义文本属性列名
text_data_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope']
labels = data['HeartDisease'].values

# 合并文本属性为一个特征向量
text_features = []
for i in range(len(data)):
    combined_text = " ".join([str(data[col][i]) for col in text_data_columns])
    text_features.append(combined_text)

# 使用词袋模型进行特征提取
vectorizer = CountVectorizer(max_features=1000)
text_features_vectorized = vectorizer.fit_transform(text_features).toarray()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(text_features_vectorized, labels, test_size=0.2, random_state=42)

# 定义模型、损失函数和优化器
input_size = text_features_vectorized.shape[1]
hidden_size = 64
output_size = 2  # 2类别的疾病: 心脏病或正常
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
accuracies = []

# 模型训练
num_epochs = 10
for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float)
    labels = torch.tensor(y_train, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 输出当前轮的损失
    train_accuracy = (torch.argmax(outputs, dim=1) == labels).float().mean().item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Train Accuracy: {train_accuracy}')

    losses.append(loss.item())
    accuracies.append(train_accuracy)

# 在测试集上评估准确率
with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float)
        test_labels = torch.tensor(y_test, dtype=torch.long)
        test_outputs = model(test_inputs)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)
        print(f'准确率: {accuracy}')

# 绘制损失值和准确率随训练次数变化的图像
plt.figure()
plt.plot(range(1, num_epochs+1), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, num_epochs+1), accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()