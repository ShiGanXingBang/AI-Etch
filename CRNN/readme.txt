pkl_file = open('featbin.pkl', 'rb')
feat = pickle.load(pkl_file)

feat.shape：(10000, 185)

每行包括四个工艺参数，181个角度的深度数据
每50行是一个刻蚀过程，过程中的工艺参数都是一样的，一共200个刻蚀过程