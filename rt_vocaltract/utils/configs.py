import torch
import yaml
import pathlib
from rt_vocaltract import models, datasets

class Configs:
    """
    Config utility functions for reproducible models.
    """

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load config file.

        Args:
            config (str): Config path
        
        Returns:
            Dictionary of configuration
        """
        config_path = pathlib.Path(config_path)
        return yaml.safe_load(config_path.read_text())

    @staticmethod
    def load_model(config: dict, ckpt: str = None, device="cpu") -> torch.nn.Module:
        """
        Load model for training or inference. Defaults to SiameseCNN with no
        model parameters.
        
        Args:
            ckpt (str): Checkpoint path
            config (dict): Configuration dict
        
        Returns:
            model (torch.nn.Module): Model based on config
        """
        model_name = config.get("model", "SoundStreamInversion")
        model_params = config.get("model_params", {})
        model = getattr(models, model_name)(**model_params)
        model.to(device)

        if ckpt is not None:
            ckpt = pathlib.Path(ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
        
        return model
    
    @staticmethod
    def load_discriminator(config: dict, ckpt: str = None, device="cpu") -> torch.nn.Module:
        disc_name = config.get("discriminator", "MelGANMultiScaleDiscriminator")
        disc_params = config.get("discriminator_params", {})
        disc = getattr(models, disc_name)(**disc_params)
        disc.to(device)

        if ckpt is not None:
            ckpt = pathlib.Path(ckpt)
            disc.load_state_dict(torch.load(ckpt, map_location=device)["discriminator_state"])

        return disc
    
    @staticmethod
    def load_criterion(config: dict, criterions: dict, arg: str = "loss"):
        """
        Load loss from dict of criterions, with L1Loss as default.

        Example:
            criterions = { "l1": torch.nn.L1Loss }

        Args:
            config (dict): Configuration dict
            criterions (dict): All possible losses dict
        """
        loss = config.get(arg, "l1")
        return criterions.get(loss)(**config.get("loss_params", {}))
    
    @staticmethod
    def load_optimizer(config: dict, model: torch.nn.Module, arg: str = "optim_params", ckpt: str = None, name: str = None, device="cpu"):
        optim_params = config.get(arg, {})
        optimizer = torch.optim.Adam(model.parameters(), **optim_params)

        if ckpt is not None:
            ckpt = pathlib.Path(ckpt)
            optimizer.load_state_dict(torch.load(ckpt, map_location=device)[name])

        return optimizer
    
    @staticmethod
    def load_scheduler(config: dict, optimizer: torch.optim.Optimizer, arg: str = "sched_params", ckpt: str = None, name: str = None, device="cpu"):
        scheduler_params = config.get(arg, {})
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)

        if ckpt is not None:
            ckpt = pathlib.Path(ckpt)
            scheduler.load_state_dict(torch.load(ckpt, map_location=device)[name])

        return scheduler
    
    @staticmethod
    def load_dataset(config: dict):
        """
        Load train, val, and test datasets based on config.

        Args:
            config (dict): Configuration dict
        """

        dataset_name = config.get("dataset", "LibriTTSRDataset")
        dataset_params = config.get("dataset_params", {})
        dataset = getattr(datasets, dataset_name)(**dataset_params)
        print(len(dataset))

        train_split = config.get("train_split", 0.8)
        val_split = config.get("val_split", 0.1)

        num_train_samples = int(len(dataset) * train_split)
        num_val_samples = int(len(dataset) * val_split)
        num_test_samples = len(dataset) - num_train_samples - num_val_samples

        (train_data, val_data, test_data) = torch.utils.data.random_split(
            dataset,
        	[
                num_train_samples, 
                num_val_samples, 
                num_test_samples
            ],
        	generator=torch.Generator().manual_seed(config.get("seed", 0))
        )

        return train_data, val_data, test_data
    
    @staticmethod
    def load_dataloader(config: dict, dataset: torch.utils.data.Dataset):
        """
        Load dataloader for a dataset based on config.

        Args:
            config (dict): Configuration dict
            dataset: Dataset to create DataLoader from
        """

        dataloader_params: dict = config.get("dataloader_params", {})

        dataset_name = config.get("dataset", "LibriTTSRDataset")
        dataset_class = getattr(datasets, dataset_name)
        collate_fn_name = dataloader_params.get("collate_fn", None)
        if type(collate_fn_name) == str:
            dataloader_params["collate_fn"] = getattr(
                dataset_class, 
                collate_fn_name
            )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            **dataloader_params
        )

        return dataloader
    
    @staticmethod
    def generate_checkpoint_dir(config: dict, val_loss: float):
        model_name = config.get("model_key", "ssinv")

        loss = config.get("loss")

        return pathlib.Path(f"{model_name}_{loss}_{val_loss:.2f}.pth")

if __name__ == "__main__":

    # Sanity Checks
    config = Configs.load_config("/data/prabhune/rt_vocaltract/ckpts/ssinv_def.yaml")

    print(config)
    print(Configs.load_model(config, "../ckpts/scnn_l1/scnn_l1_0.15.pth"))

    criterions = {
        "l1": torch.nn.L1Loss
    }
    print(Configs.load_criterion(config, criterions))

    dataset = Configs.load_dataset(config)
    print(Configs.load_dataloader(config, dataset))

