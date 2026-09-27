# raw data
""" 
raw data : {
    train : 1650(图种类) * 10(每种数量) * 4(出现次数) * 10(人数) * (63(EEG通道) , 1000(采样率) * 1(时间1000ms)) 
    test : 200(图种类) * 1 * 80 * 10 * (63, 1000)
}
"""

# preprocessed data
"""
一. 带通滤波 : 只留0.1Hz ~ 100Hz的信号
二. 基线校准 : 逐通道减去该通道 -200 ~ 0ms 的平均值
三. z-score 归一化 : z = (x - 均值) / 标准差
四. PCA 降到 256 维 : PCA 到 256 维（解释 >80% 方差）
"""





from src.api.utils.high_pass_filtering import high_pass_filltering
import numpy as np
def test_data_return(data) -> np.ndarray:
    if data == None:
        data = np.random.random([30, 63, 1200])     # 30个样本

    # preprocessed 

    # 带通滤波
    data = high_pass_filltering(data)

    # 基线校准

    embedding_data = np.mean(data[:, :, :200], axis=-1)
    x, y = embedding_data.shape

    data = data[:, :, 200:] - embedding_data.reshape(x, y, 1)



    return  data 


if "__main__" == __name__:
    print(test_data_return(None).shape)












if "__main__" == __name__:
    data = np.random.random([30, 63, 1200])







