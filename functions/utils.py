import datetime
import numpy as np
from typing import Union

def snapshot_maker(param_dict, dir:str):
    # record <.pth> model infomation snapshot.
    with open(dir, 'w') as file:
        for key, value in param_dict.items():
            file.write(key + ' : ' + str(value) + '\n')
        time_now = datetime.datetime.now()
        file.write('record time : ' + time_now.strftime('%Y-%m-%d %H:%M:%S'))


def write_line(dict_in:dict, dir:str):
    # record loss in real time.
    import os
    import torch
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    with open(dir, 'a') as file:
        for key, value in dict_in.items():
            if isinstance(key, torch.Tensor):
                key = float(key)
            if isinstance(value, torch.Tensor):
                value = float(value)
            if isinstance(key, float):
                key = round(key, 4)
            if isinstance(value, float):
                value = round(value, 6)
            file.write(str(key) + ' : ' + str(value) + '\n')


def cuda2np(tensor) -> np.ndarray:
    # cuda tensor -> cpu numpy
    arr = tensor.cpu()
    arr = arr.detach().numpy()
    return arr


def tensorview(Intensor, batch_idx):
    # show target tensor
    arr = cuda2np(Intensor)
    print(arr[batch_idx])


def imgstore(Intensor, nums:int, save_dir:str, epoch:Union[int, str], filename='', cls='pred'):
    # function for saving prediction image.
    import os
    import cv2

    img_np = cuda2np(Intensor)
    img_np = np.transpose(img_np, (0, 2, 3, 1))

    os.makedirs(save_dir, exist_ok=True)

    img_list = []

    for i, img in enumerate(img_np):
        if i == nums:
            break
        img_list.append(img)

    if isinstance(filename, str):  # stores only one image, batch == 1
        if isinstance(epoch, str):
            cv2.imwrite(os.path.join(save_dir, cls + '_' + epoch + '_[' + filename + '].png'), img_list[0])
        else:
            cv2.imwrite(os.path.join(save_dir, cls+'_'+'epoch_'+str(epoch)+'_['+filename+'].png'), img_list[0])

    elif isinstance(filename, list):  # stores <nums:int> images, batch > 1
        for idx, unit in enumerate(img_list):
            if isinstance(epoch, str):
                cv2.imwrite(os.path.join(save_dir, cls + '_' + epoch + '_[' + filename[idx] + '].png'), unit)
                print(f"{os.path.join(save_dir, cls+'_'+epoch+'_['+filename[idx]+'].png')} saved.")
            else:
                cv2.imwrite(os.path.join(save_dir, cls+'_'+'epoch_'+str(epoch)+'_['+filename[idx]+'].png'), unit)