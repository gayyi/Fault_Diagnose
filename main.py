from collections import Counter
from communication import save_first_last_files, detect_new_files
from decode import tim_to_data, get_sample
import numpy as np
import tkinter as tk
#from tkinter import ttk
import tkinter.font as tkFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import time
import sys
import os
import torch
# 假设这里有communication, detect_new_files, tim_to_data, get_sample的相关代码
model_path = "C:\\Users\\lijingyidelianxiang\\Desktop\\Communication\\demo\\saved_models\\2-CNN_2conv_2fc-361-0.972.pth"
last_seen_files_dap = []  # Last detected file list, initially empty
last_seen_files_tim = []
new_file_dap = []
new_file_tim = []
model = torch.load(model_path, map_location=torch.device('cpu'))
arr = None
cls = None  # No longer create StringVar object here
is_diagnosing = False  # Flag variable to control the diagnostic process
validity = 0 #here for test the validity of input data

def find_max_column_sum(arr):
    column_sums = np.sum(arr, axis=0)
    max_sum_index = np.argmax(column_sums)
    max_sum_value = column_sums[max_sum_index]
    return max_sum_index, max_sum_value 

def get_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

def check_array_validity(arr,threshold):
    """test the validity of the given arr"""
    #arr = tim_to_data()
    if np.any(arr > threshold):
        return 'This array is valid'
    else:
        return 'This array is invalid'
    
def update_plot(frame):
    global arr
    if arr is not None:
        ax.clear()
        ax.plot(np.arange(len(arr)), arr, label='Vibration Signal')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        canvas.draw()

def update_cls():
    global arr
    if arr is None:
        cls.set("没有检测输入")
    # 如果arr不为None，则不需要改变cls的值，因为它将在诊断过程中被更新
    root.after(1000, update_cls)


def periodic_task():
    global is_diagnosing
    if is_diagnosing:
        time.sleep(10)
        global new_file_dap, new_file_tim, arr
        input_folder = "C:\\Users\\lijingyidelianxiang\\Desktop\\Communication\\input_folder\\2024年\\3月"
        output_file = "C:\\Users\\lijingyidelianxiang\\Desktop\\Communication\\output_folder"
        save_first_last_files(input_folder, output_file)

        target_folder_dap = "C:\\Users\\lijingyidelianxiang\\Desktop\\Communication\\output_folder\\dap_files"
        target_folder_tim = "C:\\Users\\lijingyidelianxiang\\Desktop\\Communication\\output_folder\\first_files"

        last_file_dap = new_file_dap
        new_file_dap = detect_new_files(target_folder_dap, last_seen_files_dap)
        last_file_tim = new_file_tim
        new_file_tim = detect_new_files(target_folder_tim, last_seen_files_tim)

        if new_file_dap != last_file_dap and new_file_dap:
            if new_file_tim != last_file_tim and new_file_tim:
                arr = tim_to_data(new_file_tim[-1], new_file_dap[-1])
                validity = check_array_validity(arr, 0.05)
                py_tensor = get_sample(arr, sp_num=8, sp_len=2048)

                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
                input_tensor = py_tensor.to(device)
                with torch.no_grad():
                    output_tensor = model(input_tensor)

                    output_np = output_tensor.cpu().numpy()
                    max_sum_index, max_sum_value = find_max_column_sum(output_np)

                if validity == 1:
                    if max_sum_index == 0:
                        cls.set('Normal')
                    elif max_sum_index == 1:
                        cls.set('Ball')
                    elif max_sum_index == 2:
                        cls.set('Inner Race')
                    elif max_sum_index == 3:
                        cls.set('Outer Race')
                else: cls.set('No valid input data')

    root.after(1000, periodic_task)  # Execute every 1000 milliseconds (1 second)

def start_diagnosis():
    global is_diagnosing
    is_diagnosing = True
    # 可能还需要添加其他启动诊断的逻辑

def stop_diagnosis():
    global is_diagnosing, arr
    is_diagnosing = False
    arr = None
    cls.set("没有检测输入")
    ax.clear()
    canvas.draw()

root = tk.Tk()
root.title("Fault Detection Tool")
# 正确初始化cls
cls = tk.StringVar(value="没有检测输入")
# 创建主框架
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# 创建故障类型显示标签
fault_type_label = tk.Label(main_frame, textvariable=cls, font=('Times', 20, 'italic'))
cls.set("没有检测输入")  # 初始化显示
fault_type_label.pack(side=tk.TOP, pady=5)

# 创建绘图框架
plot_frame = tk.Frame(main_frame)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# 创建按钮框架
buttons_frame = tk.Frame(root)
buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

ftbutton = tkFont.Font(family='Times', size=20, slant=tkFont.ITALIC)
start_button = tk.Button(buttons_frame, text="开始检测", font=ftbutton, command=start_diagnosis, height=2, width=20)
start_button.pack(side=tk.LEFT, expand=True, padx=10, pady=5)

stop_button = tk.Button(buttons_frame, text="停止检测", font=ftbutton, command=stop_diagnosis, height=2, width=20)
stop_button.pack(side=tk.RIGHT, expand=True, padx=10, pady=5)

# 创建实时更新图表的动画
ani = FuncAnimation(fig, update_plot, frames=np.arange(100), interval=1000)

update_cls()  # 开始更新故障类型显示
periodic_task()  # 开始周期性任务

root.mainloop()