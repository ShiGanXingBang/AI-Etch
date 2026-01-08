import numpy as np
import pickle
import glob
import pandas as pd
import os

# 1. 设定你的 CSV 文件夹路径 (Week12/13 生成的那些 CSV)
CSV_FOLDER = r"E:\MachineLearning\data\py\Monte_Carlo\Monte_Carlo\Csv\Test_Output"

# 2. 设定你的工艺参数 (假设这一批 CSV 都是在这个参数下跑的)
# 如果你有不同的参数，需要根据文件名或者日志来动态读取
# 这里举例：[功率, 气压, 流量, 偏压]
current_params = [500, 20, 50, 100] 

all_data = []

# 3. 获取所有 CSV 文件，按顺序读取
# 假设每个 CSV 代表一个时刻，文件名里带有步数，比如 contour_step_0.csv
file_list = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))

# 假设我们要凑齐 200 组实验，每组 50 个时刻，这里简化逻辑，只演示如何打包
# 你需要确保你的 CSV 是按 (实验1_时刻0...时刻49, 实验2_时刻0...时刻49) 顺序排列的

print(f"找到 {len(file_list)} 个文件，开始转换...")

for file_path in file_list:
    try:
        # 读取 CSV，假设第一列是 X，第二列是 Y
        df = pd.read_csv(file_path)
        
        # 提取 181 个点的 Y 坐标 (假设你的网格宽度对应 181 个点)
        # 如果你的点数不是 181，需要插值或者降采样！
        # 这里假设 CSV 里正好有 181 行数据
        if len(df) != 181:
            # 如果不是181，强行插值成181个点 (为了匹配神经网络输入)
            y_raw = df.iloc[:, 1].values
            x_raw = np.arange(len(y_raw))
            y_interp = np.interp(np.linspace(0, len(y_raw)-1, 181), x_raw, y_raw)
            contour_data = y_interp
        else:
            contour_data = df.iloc[:, 1].values # 取第2列(Y坐标)

        # 4. 拼接：轮廓数据 (181) + 物理参数 (4)
        # 结果是一个长度为 185 的向量
        combined_row = np.concatenate((contour_data, current_params))
        
        all_data.append(combined_row)
        
    except Exception as e:
        print(f"文件 {file_path} 读取失败: {e}")

# 5. 转换为 Numpy 数组
final_feat = np.array(all_data)
print(f"最终数据形状: {final_feat.shape}") 
# 期望形状: (N, 185)

# 6. 保存为 PKL
with open('featbin.pkl', 'wb') as f:
    pickle.dump(final_feat, f)

print("转化完成！生成的 featbin.pkl 可以直接喂给 main.py 了！")