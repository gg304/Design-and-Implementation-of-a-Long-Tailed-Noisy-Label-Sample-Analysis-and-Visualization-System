项目简介
本系统是一个面向长尾噪声标签数据的可视化分析工具。系统参考TABASCO方法的核心思想，通过计算加权JS散度（WJSD）和自适应质心距离（ACD）两个度量，对样本进行两阶段选择，将样本分为高可信、低可信和疑似噪声三类。系统支持CIFAR-10N真实噪声数据集和CIFAR-10-LT模拟数据集，提供长尾分布可视化、噪声检测分析、头尾部类别对比、交互式标签修正等功能。

系统功能
功能模块	说明
数据集加载	支持CIFAR-10N（真实噪声）、CIFAR-100N、Animal-10N、CIFAR-10-LT（模拟）
TABASCO检测	两阶段样本选择，输出高可信/低可信/疑似噪声三类
可视化分析	类别分布图、混淆矩阵、置信度分布、特征空间PCA可视化
交互修正	支持逐样本标签修改、批量修正、结果导出CSV
头尾对比	自动划分头尾部类别，对比准确率、噪声率等指标
环境要求
依赖	版本
Python	3.9+
PyTorch	2.0+
Streamlit	1.25+
NumPy	1.24+
Pandas	2.0+
Plotly	5.14+
scikit-learn	1.3+
安装步骤
bash
# 1. 克隆或下载项目
cd H:\PY\Project

# 2. 安装依赖
pip install streamlit torch torchvision numpy pandas matplotlib plotly scikit-learn scipy pillow requests

# 3. 准备数据集（可选）
# 将 CIFAR-10_human_ordered.npy 放入 ./data/ 目录
运行程序
bash
streamlit run main.py
浏览器会自动打开 http://localhost:8501

数据集说明
CIFAR-10N（真实噪声）
噪声类型	噪声率	说明
aggregate	约9%	多人标注投票结果
random1/2/3	约17-18%	随机选取单人的标注
worst	约40%	故意选取错误标注
CIFAR-10-LT（模拟）
参数	说明
类别数量	2-50类
长尾不平衡率	0.05-0.5
噪声率	0%-50%
使用流程
选择数据集：侧边栏选择数据集类型

设置参数：样本数量、噪声类型等

加载数据：点击加载/生成按钮

运行检测：选择模型，点击TABASCO检测

查看结果：浏览四个标签页

修正标签：在标签修正页修改疑似噪声

导出结果：下载CSV文件

项目结构
text
H:\PY\Project\
│
├── main.py                 # 主程序
├── data/                   # 数据集目录
│   ├── cifar-10-python.tar.gz
│   └── CIFAR-10_human_ordered.npy
└── torch_cache/            # 模型缓存
    └── hub/checkpoints/
        ├── resnet18-f37072fd.pth
        ├── resnet34-b627a593.pth
        └── resnet50-0676ba61.pth
常见问题：
问题	解决方法
数据集下载	https://www.cs.toronto.edu/~kriz/cifar.html可从该网址选取下载放入 ./data/ 目录
模型权重不存在	手动下载放入 ./torch_cache/hub/checkpoints/
页面卡顿	减少样本数量（建议500-1000）
整体准确率低	使用模拟数据测试，或在次基础上进行微调模型或重训练模型

输出文件：
导出CSV包含以下列
列名	说明
样本ID	样本唯一标识
真实标签	真实类别（clean_label）
原始噪声标签	加载时的噪声标签
修改后标签	用户修正后的标签
预测标签	模型预测的类别
置信度	预测置信度
样本类型	高可信/低可信/疑似噪声
原始是否为噪声	原始标签是否错误
修改是否正确	修正后是否正确
