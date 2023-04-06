import os
import wave

# 定义要搜索的目录路径
folder_path = "F:\\yhc\\ABCS\\ABCS_database_ciaic\\ABCS_database\\train\\air"

# 遍历目录中的所有文件
for file_name in os.listdir(folder_path):
    # 如果文件名以.wav结尾，说明是音频文件
    if file_name.endswith(".wav"):
        # 用wave模块打开音频文件
        with wave.open(os.path.join(folder_path, file_name), "rb") as wave_file:
            # 获取音频文件的采样率
            frame_rate = wave_file.getframerate()
            # 获取音频文件的采样点数
            frame_count = wave_file.getnframes()
            # 如果采样率是16k，并且采样点数小于16384，说明音频文件太短，需要删除
            if frame_rate == 16000 and frame_count < 32000:
                # 删除音频文件，并打印提示信息
                wave_file.close()
                # os.remove(os.path.join(folder_path, file_name))
                print(f"Deleted {file_name} because it has only {frame_count} samples.")

# Deleted Speaker47_D_110.wav because it has only 14562 samples.
# Deleted Speaker47_D_176.wav because it has only 13923 samples.
# Deleted Speaker47_D_289.wav because it has only 15365 samples.
# Deleted Speaker47_D_97.wav because it has only 14082 samples.
# Deleted Speaker54_D_266.wav because it has only 14896 samples.
# Deleted Speaker62_D_1.wav because it has only 5440 samples.
# Deleted Speaker62_D_283.wav because it has only 16055 samples.
# Deleted Speaker66_C_116.wav because it has only 15840 samples.
# Deleted Speaker66_C_47.wav because it has only 14880 samples.
# Deleted Speaker66_C_67.wav because it has only 15840 samples.
# Deleted Speaker66_D_238.wav because it has only 15641 samples.
# Deleted Speaker86_D_3.wav because it has only 15973 samples.
# Deleted Speaker99_C_161.wav because it has only 13410 samples.
# Deleted Speaker5_C_65.wav because it has only 8640 samples.
# 进程已结束,退出代码0
