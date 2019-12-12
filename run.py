import os

for num in range(1):
    print("Running camera {} with GPU {}".format(num, num))
    os.system("CUDA_VISIBLE_DEVICES={} python detect.py --wecbam {}".format(num, num))