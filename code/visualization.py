import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data_handler
import matplotlib.colors as colors

t = data_handler.CsvHandler("APPL")

# 创建年份和交易日的数据
df = t.get_equal_length_prices()
norm_data = []
for i in range(len(t.years)):
    tmp = []
    for j in range(t.max_days):
        tmp.append(df[2018 + i][j])
    norm_data.append(tmp)

# 转换为numpy数组
norm_data = np.array(norm_data)

years = np.array(t.years)
trading_days = np.arange(0, t.max_days)

# 使用更漂亮的主题和增加分辨率
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制z=0的平面，降低绿色平面的透明度
Y, X = np.meshgrid(trading_days, years)
Z = np.zeros(X.shape)
ax.plot_surface(X, Y, Z, color='green', alpha=0.1)

# 创建两个颜色映射，一个用于年份，另一个用于正负值
color_map_years = cm.get_cmap('viridis', len(years))
color_map_values = cm.get_cmap('RdBu', 8)

# 对每一年的数据分别绘制一条线
for i in range(len(years)):
    for j in range(len(trading_days)):
        # 根据z值设置颜色，降低颜色映射的上限
        if norm_data[i, j] < 0:
            color_value = cm.Blues(abs(norm_data[i, j]) / 8)
        else:
            color_value = cm.Reds(norm_data[i, j] / 8)
        color_year = color_map_years(i)

        # 在交易日上增加一个偏移量，以防止数据重叠
        offset = (i - len(years) / 2) * 10
        ax.scatter(years[i], trading_days[j] + offset, norm_data[i, j], color=color_year, s=5)

# 设置坐标轴标签和标题的字体大小
ax.set_xlabel('Years', fontsize=8)
ax.set_ylabel('Trading Days', fontsize=8)
ax.set_zlabel('Normalized Data', fontsize=8)

# 显示图像
plt.show()
