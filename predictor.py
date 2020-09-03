import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

from parameters import *
from networks.nets import *
from dataloader.dataset import MetaDataset
from dataloader import dict_transforms
import functions.utils as util

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    branch_num = 1
    epoch_num = 3
    # ========================================== dir & param ==========================================
    data_dir = r'/home/user/Desktop/test_dataset'
    weight_dir = os.path.join('save', 'branch_'+str(branch_num), model_name+'_epoch_'+str(epoch_num)+'.pth')
    dst_dir = os.path.join(data_dir, 'predicton/branch_1')
    store_num = 10
    the_name = os.path.splitext(os.path.basename(weight_dir))[0]
    assert test_params['test_batch'] >= store_num, 'batch size must be bigger than the number of storing image.'
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== transform ===========================================
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_set = transforms.Compose([
        transforms.Resize(size=(250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ============================================ Dataset ============================================
    test_set = ImageFolder(root='./data/test', transform=transform_set)
    print(f'training set classes : {test_set.classes}')
    print(f'test data : {len(test_set)} files detected.')
    test_loader = DataLoader(dataset=test_set, batch_size=test_params['test_batch'],
                              shuffle=False, num_workers=user_setting['test_processes'])
    classes = {0:'masked', 1:'unmasked'}
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # =========================================== Model Load ==========================================
    netend = Classifier(2)
    model = ResNet50(netend, 2)
    model.load_state_dict(torch.load(weight_dir))
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

    # ========================================== GPU setting ==========================================
    environment = {}
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'GPU {torch.cuda.get_device_name()} available.')
        model.cuda()
        environment['gpu'] = True

    else:
        device = torch.device('cpu')
        print(f'GPU unable.')
        environment['gpu'] = False
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------
    os.makedirs(dst_dir, exist_ok=True)
    print(f'save directory : {dst_dir}')
    # ============================================ run ================================================
    model.eval()
    for i, data in enumerate(test_loader):
        image, label = data

        if environment['gpu']:
            image = image.cuda()

        with torch.no_grad():
            output = model.forward(image)

        value, indices = output.max(-1)
        for i, val in indices, value:
            print(f'predict {classes[i]} with {val*100:.2f} %.')

        print(f'{output.max} / {label}')
    # =================================================================================================

    # ------------------------------------------------------------------------------------------------------------------

