import os
import re
import time
import pandas as pd
import torch

from .utils import config as cg
from .utils import config_util as cu
from .utils import survival as surv
from .segmentation import preproc_pipeline as PPP


def choose_models(cfg):
    if 'resCRNN' in cfg["model_type"]:
        from .CNN.resRNN import resCRNN_n
        return resCRNN_n(**cu.get_model_args(cfg))
    elif 'denseCRNN' in cfg["model_type"]:
        from .CNN.denseRNN import denseCRNN_n
        return denseCRNN_n(**cu.get_model_args(cfg))


def select_model_path(outcome, model_path=None):
    if model_path is None:
        model_path = os.path.dirname(__name__)+"../models/"
    path = model_path+"/"+outcome+"_model.pt"
    return path


def proc_CNN_out(CNN_out, cf):
    output = surv.output2surv(CNN_out.cpu(), cf)
    out = list(output.values.reshape(-1))
    one_dict = {}
    for index, i in enumerate(out):
        key = "{} month".format(index*6)
        one_dict[key] = i
    return one_dict


class FICHnet_model:
    def __init__(self, outcome, img_paths, model_path=None, device="cpu"):
        '''
        Args:
            `outcome`: long-term post-ICH outcome to predict, can be
            "disability", "severe disability" or "dependent"
            `img_paths`: can be the path to a single nifti file, or a text file
            containing absolute image paths
        '''
        self.outcome = outcome

        if ".nii.gz" in img_paths:  # for one case
            self.ipaths = [img_paths]
        else:  # for a list of cases stored in a txt file
            with open(img_paths, "r") as f:
                self.ipaths = f.readlines()
                self.ipaths = [re.sub("\\n", "", i) for i in self.ipaths]

        if outcome != "severe_disability":
            model_type = "resCRNN34"
        else:
            model_type = "denseCRNN201"

        self.model_path = select_model_path(self.outcome, model_path)
        new_cf = cg.config.copy()
        new_cf["model_type"] = model_type
        new_cf["device"] = device
        out_dict = {"dependent": "Dependent", "disability": "Disability",
                    "severe_disability": "Disability5"}
        new_cf["outcome_col"] = out_dict[outcome]
        self.new_cf = cu.postprocess(new_cf)

    def __predict__(self, model, img_path):
        start = time.time()
        img = PPP.preprocess(img_path)

        if img is not None:
            device = self.new_cf["device"]
            img = torch.tensor(img, device=device, dtype=torch.float32)
            img = img.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            preproc_time = time.time() - start

            with torch.no_grad():
                outputs = model(img)
            out = proc_CNN_out(outputs, self.new_cf)
            out["time_total"] = time.time() - start
            out["time_preproc"] = preproc_time
            return out
        else:
            return None

    def predict(self):
        device = self.new_cf["device"]
        model = choose_models(self.new_cf)
        model.to(device)
        model.load_state_dict(torch.load(self.model_path,
                              map_location=torch.device(device)))
        model.eval()

        out_dict = {}
        for i in self.ipaths:
            ID = re.sub(".nii.gz$", "", os.path.basename(i))
            out = self.__predict__(model, i)
            if out is not None:
                out_dict[ID] = out
            else:
                print("{} has incomplete brain coverage".format(ID))
        out_dict = pd.DataFrame(out_dict)
        return out_dict


def predict_all(img_paths, model_path, save_dir, device="cpu"):
    '''
    Args:
        `img_paths`: path to a text file containing absolute path of all the
        images
        `model_path`: path containing model weight, with the following files:
            dependent_model.pt
            disability_model.pt
            severe_disability_model.pt
        `save_dir`: the model will output the following files:
            pred_dependent.csv
            pred_disability.csv
            pred_severe_disability.csv
    '''
    all_vars = ["dependent", "disability", "severe_disability"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in all_vars:
        one_model = FICHnet_model(i, img_paths, model_path, device)
        out_df = one_model.predict()
        out_df.to_csv(save_dir+"/pred_"+i+".csv")
        del one_model
        torch.cuda.empty_cache()
