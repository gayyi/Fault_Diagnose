import os
import shutil
import time
from collections import Counter

from decode import tim_to_data, get_sample
from model import *
from config import Config


def clear_proj(proj):
    for item in os.listdir(proj):  # 遍历proj下每一个文件（文件夹）
        if not item.endswith('prj'):  # 如果不是.prj文件
            item_cont = os.listdir(os.path.join(proj, item))  # 获取该item下所有文件
            if len(item_cont) != 5:  # 如果item下文件内容小于5个（四个通道+dap）
                shutil.rmtree(os.path.join(proj, item))

# proj为项目所在目录，root是当前所分析的数据所在目录，分析完毕后被删除
# 输入项目目录，输出该项目下的一个dir（应保证仅有一个dir，用完即删） root=proj+dir
def get_root_dir(proj='test_data/2024年/1月'):
    root = ''
    tgt_dir = ''
    for idx, dir in enumerate(os.listdir(proj)):
        if os.path.isdir(os.path.join(proj, dir)):
            root = os.path.join(proj, dir).replace('\\', '/')
            tgt_dir = dir
    return root, tgt_dir


# 输入为：模型输出的张量，输出为预测的故障类型
def get_cls(input_ts):

    l = input_ts.cpu().numpy().tolist()
    counter = Counter(l)
    max_count = max(counter.values())
    result = [num for num, count in counter.items() if count == max_count]

    if result == 0:
        cls = 'Normal'
    elif result == 1:
        cls = 'Ball'
    elif result == 2:
        cls = 'Inner Race'
    elif result == 3:
        cls = 'Outer Race'
    else:
        cls = 'Abnormal Result'

    return cls


while True:
    proj = 'test_data/2024年/1月'
    root, dir = get_root_dir(proj)
    #print(root)

    #time.sleep(0.5)

    # 如果项目下非空，则执行推理任务
    if dir != '' and len(os.listdir(proj)) == 2 and len(os.listdir(root)) == 5:
        print(dir)

        '''获取tim和dap路径'''
        channel = 1  # 确定提取的通道
        tim_pth = os.path.join(root, dir + f'#1-{channel}.tim').replace('\\', '/')
        dap_pth = os.path.join(root, dir + '.dap').replace('\\', '/')

        '''获取输入张量数据'''
        data = tim_to_data(tim_pth, dap_pth)
        inputs = get_sample(data, 1, 2048).to(Config.device)  # 数据转换，从.tim中得到输入tensor

        '''推理并输出预测结果'''
        model = CNN_2conv_2fc().to(Config.device)
        outputs = model(inputs)
        print(outputs.cpu().detach().numpy().tolist())
        results = outputs.argmax(1)
        fault_cls = get_cls(results)
        print(fault_cls)

        '''删除当前目录'''
        shutil.rmtree(root)

    else:
        pass
