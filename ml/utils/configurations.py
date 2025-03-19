import yaml
import multiprocessing
import torch
import math
from rdkit.Chem.Draw import MolDrawOptions
from utils.logging_deepscreen import logger
from ray import tune


class Configurations:
    def __init__(self, config_path="/root/trypanodeepscreen/config/config.yml"):
        self.load_config(config_path)
        logger.debug("Configuration class instantiated with YAML file")

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        logger.debug(f"Configuration file {config_path} loaded.")

    def get_mol_draw_options(self):
        opt = MolDrawOptions()
        opt.atomLabelFontSize = self.config["mol_draw_options"]["atomLabelFontSize"]
        opt.dotsPerAngstrom = self.config["mol_draw_options"]["dotsPerAngstrom"]
        opt.bondLineWidth = self.config["mol_draw_options"]["bondLineWidth"]
        return opt
    
    def get_img_size(self):
        img_size = self.config["img_size"]
        return (img_size, img_size)

    def get_use_tmp_imgs(self):
        return self.config["use_tmp_imgs"]
    
    def get_hyperparameters_search(self):
        return {
            'fully_layer_1': tune.choice(self.config["hyperparameters_search"]["fully_layer_1"]),
            'fully_layer_2': tune.choice(self.config["hyperparameters_search"]["fully_layer_2"]),
            'learning_rate': tune.choice(self.config["hyperparameters_search"]["learning_rate"]),
            'batch_size': tune.choice(self.config["hyperparameters_search"]["batch_size"]),
            'drop_rate': tune.choice(self.config["hyperparameters_search"]["drop_rate"]),
        }
    
    def get_hyperparameters_search_setup(self):
        return self.config["hyperparameters_search_setup"]
    
    def get_raytune_scaleing_config(self):
        if self._get_gpu_number() > 0:
            return {
                "num_workers": min(self._get_gpu_number(), self.config.get("max_gpus", self._get_gpu_number())),
                "use_gpu": True,
                "resources_per_worker": {"CPU": self._get_cpu_number(), "GPU": 1}
            }
        else:
            return {
                "num_workers": 1,
                "use_gpu": False,
                "resources_per_worker": {"CPU": self._get_cpu_number()}
            }
    
    def get_data_splitting_config(self):
        return self.config["data_splitting"]
    
    def _get_cpu_number(self):
        cores = min(multiprocessing.cpu_count(), self.config.get("max_cpus", multiprocessing.cpu_count()))
        gpus = self._get_gpu_number()
        if gpus > 0:
            return min(gpus * 4, cores) if gpus * 4 <= cores else math.floor(cores / gpus)
        return cores

    def _get_gpu_number(self):
        return min(torch.cuda.device_count(), self.config.get("max_gpus", torch.cuda.device_count()))
        

configs = Configurations()
