# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import typing

from cog import BasePredictor, Path, Input, File
import shutil
import os

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import torch
from torch.autograd import Variable
from u2net_test import normPRED, save_output
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset


class Predictor(BasePredictor):
    net = None

    def setup(self):
        model_name = 'u2net'  # u2netp
        model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

        if model_name == 'u2net':
            print("...load U2NET---173.6 MB")
            net = U2NET(3, 1)
        elif model_name == 'u2netp':
            print("...load U2NEP---4.7 MB")
            net = U2NETP(3, 1)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()

        self.net = net

    def predict(
        self,
        image: Path,
    ) -> Path:
        prediction_dir = "outputs/"
        shutil.rmtree(prediction_dir, ignore_errors=True)

        image = str(image)
        img_name_list = [image]

        # --------- 2. dataloader ---------
        # 1. dataloader
        test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                            lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):

            print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7 = self.net(inputs_test)

            # normalization
            pred = d1[:, 0, :, :]
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test], pred, prediction_dir)

            del d1, d2, d3, d4, d5, d6, d7

        return Path(prediction_dir + os.listdir(prediction_dir)[0])
