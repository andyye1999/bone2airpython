import os

path1 = '/home/dsp/yhc/bone/test/'  # need to enhance dir


path_write = 'ceshi.txt' # txt文件写入路径(包含txt文件名)

files1 = os.listdir(path1)

print(files1)


with open(path_write, 'w+') as txt:
    for i in range(len(files1)):
        string = path1 + files1[i]
        print('string', string)
        txt.write(string + '\n')

txt.close()