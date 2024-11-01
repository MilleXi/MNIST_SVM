import sys
import time
import svm
import os
import joblib


path= sys.path[0]
model_path=os.path.join(path,r'svm.model')
path=sys.path[0]
tbasePath = 'mnist_test'
tst = time.process_time()
clf = joblib.load(model_path)
testPath = tbasePath


tflist = svm.get_file_list(testPath)
tdataMat,tdataLabel = svm.read_and_convert(tflist)
print("测试集数据维度为:{0}， 标签数量: {1}".format(tdataMat.shape,len(tdataLabel)))


score_st=time.process_time()
score = clf.score(tdataMat, tdataLabel)
score_et=time.process_time()
print("计算准确率花费 {:.6f}秒.".format(score_et - score_st))
print("准确率: {:.6f}.".format(score))
print("错误率:{:.6f}.".format((1 - score)))
tet = time.process_time()
print("测试总耗时{:.6f}秒.".format(tet - tst))