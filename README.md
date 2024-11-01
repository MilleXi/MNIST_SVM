# cv_exp4
1. git 加载项目文件
```
git clone https://github.com/MilleXi/cv_exp4.git
```bash
2. 安装依赖
```
pip install -r requirements.txt
```bash
3. 生成数据集
```
python get_test_data.py
python get_train_data.py
```bash
4. 训练模型
```
python svm.py
```bash
5. 评估模型
```
python svmtest.py
```bash
6. 运行web可视化框架
```
python user_interface.py
```bash
7. 打开终端中弹出的网址（注意不能开梯子）即可完成可视化操作