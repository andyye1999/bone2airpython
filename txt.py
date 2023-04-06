import os

path1 = 'F:\\yhc\\bone\\Bone_Air_LCH\\bone\\train2s\\' # 带噪语音路径
path2 = 'F:\\yhc\\bone\\Bone_Air_LCH\\air\\train2s\\' # 干净语音路径

path_write = 'bone2strainset.txt' # txt文件写入路径(包含txt文件名)

files1 = os.listdir(path1)
# files1.sort()
files2 = os.listdir(path2)
# files2.sort()
print(files1)
print(files2)

with open(path_write, 'w+') as txt:
    for i in range(len(files1)):
        string = path1 + files1[i] + ' ' + path2 + files2[i]
        print('string', string)
        txt.write(string + '\n')

txt.close()