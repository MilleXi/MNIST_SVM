# MNIST_SVM

1. git 加载项目文件

```bash
git clone https://github.com/MilleXi/MNIST_SVM.git
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 生成数据集

```bash
python get_test_data.py
python get_train_data.py
```

4. 训练模型

```bash
python svm.py
```

5. 评估模型

```bash
python svmtest.py
```

6. 运行 web 可视化框架

```bash
python user_interface.py
```

7. 打开终端中弹出的网址（注意不能开梯子）即可完成可视化操作
