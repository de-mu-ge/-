def high_pass_filltering(data):
    import numpy as np
    from scipy.signal import butter, filtfilt

    # =========================
    # 1. 假设你的数据
    # =========================
    # 形状: (trials, channels, time)
    # data = np.random.randn(30, 63, 12000)

    fs = 1000          # 采样率，按论文设 1000 Hz
    lowcut = 0.1       # 高通截止
    highcut = 100.0    # 低通截止
    order = 4          # Butterworth 阶数

    # =========================
    # 2. 设计 Butterworth 带通滤波器
    # =========================
    # 注意: scipy 的 butter 要求归一化频率，范围 (0, 1)，1 对应 fs/2
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    # =========================
    # 3. 零相位滤波
    # =========================
    # filtfilt 沿时间轴 (axis=-1) 滤波
    data = filtfilt(b, a, data, axis=-1)


    return data

if "__main__" == __name__:
    import numpy as np

    data = np.random.random([30,63, 1200])

    filtered = high_pass_filltering(data)

    print("原始数据形状:", data.shape)
    print("滤波后形状:", filtered.shape)

    # =========================
    # 4. 简单检查
    # =========================
    print("原始数据 std:", data.std())
    print("滤波后 std:", filtered.std())


    # 原始数据形状: (30, 63, 1200)
    # 滤波后形状: (30, 63, 1200)
    # 原始数据 std: 0.2886844944273927
    # 滤波后 std: 0.3386396343288748