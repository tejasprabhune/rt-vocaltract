import argparse
import pathlib
from tqdm import tqdm

import torch

from rt_vocaltract.models import SoundStreamInversion
from rt_vocaltract.datasets import LibriTTSRDataset
from rt_vocaltract.utils.configs import Configs
from rt_vocaltract.losses import DiscriminatorAdversarialLoss, GeneratorAdversarialLoss

class Trainer():
    """
    Configurable trainer
    """
    def __init__(
        self,
        epochs,
        train_dataloader,
        val_dataloader,
        model,
        discriminator,
        ema_loss,
        gen_adv_loss,
        disc_adv_loss,
        optimizer,
        disc_optimizer,
        scheduler,
        disc_scheduler,
        checkpoint_interval,
        config,
        config_dir,
        device=torch.device("cpu")
    ):
        """
        Initializes Trainer for the specified
        model and configuration.

        Args:
            epochs: Number of epochs for training
            train_dataloader: Dataloader with all 
                training batches
            val_dataloader: Dataloader with all 
                validation batches
            model: Model to train
            discriminator: Discriminator to train
            criterion: Loss to use when training
            optimizer: Optimizer for backprop
            checkpoint_inverval: Interval of epochs
                to save a checkpoint after
            config: Configs object with model
                and dataset configuration information
            config_dir: Directory where config is stored,
                used for checkpoint saving
            device: CUDA device to train with
        """
        self.train_steps = len(train_dataloader)
        self.val_steps = len(val_dataloader)
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.discriminator = discriminator
        self.ema_loss = ema_loss
        self.gen_adv_loss = gen_adv_loss
        self.disc_adv_loss = disc_adv_loss
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.scheduler = scheduler
        self.disc_scheduler = disc_scheduler
        self.checkpoint_interval = checkpoint_interval
        self.config = config
        self.config_dir = pathlib.Path(config_dir)
        self.device = device

        self.model.to(self.device)
        self.discriminator.to(self.device)
        self.ema_loss.to(self.device)

    def train(self):
        epoch_tqdm = tqdm(range(self.epochs))
        for epoch in epoch_tqdm:
            epoch_tqdm.set_description(f"Epoch {epoch}")

            total_train_loss, total_train_disc_loss = self.train_loop(epoch)

            avg_train_loss = total_train_loss / self.train_steps
            avg_train_disc_loss = total_train_disc_loss / self.train_steps
            epoch_tqdm.set_postfix(train_loss=avg_train_loss, disc_loss=avg_train_disc_loss)

            if epoch % self.checkpoint_interval == 0:
                total_val_loss, total_val_disc_loss = self.val_loop()
                avg_val_loss = total_val_loss / self.val_steps
                avg_val_disc_loss = total_val_disc_loss / self.val_steps

                print(f"Train Loss: {avg_train_loss:.2f}")
                print(f"Train Disc Loss: {avg_train_disc_loss:.2f}")
                print(f"Val Loss: {avg_val_loss:.2f}")
                print(f"Val Disc Loss: {avg_val_disc_loss:.2f}")

                self.save_checkpoint(avg_val_loss)
            
    def train_loop(self, epoch):
        self.model.train()
        total_train_loss = 0
        total_train_disc_loss = 0

        train_tqdm = tqdm(enumerate(self.train_dataloader))
        train_tqdm.set_description(f"Epoch {epoch}")
        for i, batch in train_tqdm:
            loss, disc_loss = self.train_step(batch)
            total_train_loss += loss
            total_train_disc_loss += disc_loss
            train_tqdm.set_postfix(loss=loss, disc_loss=disc_loss)

            if i % self.checkpoint_interval == 0:
                self.save_checkpoint(loss)
        
        return total_train_loss, total_train_disc_loss
    
    def train_step(self, batch):
        loss, disc_loss = self.step(batch)

        return loss.item(), disc_loss.item()

    @torch.no_grad()
    def val_loop(self):
        self.model.eval()
        total_val_loss = 0
        total_val_disc_loss = 0
        val_tqdm = tqdm(self.val_dataloader)
        val_tqdm.set_description("Validation")

        for batch in val_tqdm:
            val_loss, disc_val_loss = self.val_step(batch)
            total_val_loss += val_loss
            total_val_disc_loss += disc_val_loss
            val_tqdm.set_postfix(val_loss=val_loss, disc_val_loss=disc_val_loss)

        return total_val_loss, total_val_disc_loss
    
    @torch.no_grad()
    def val_step(self, batch):
        return tuple(map(lambda x: x.item(), self.step(batch)))

    def step(self, batch):
        return self.disc_step(batch), self.gen_step(batch)

    def gen_step(self, batch):
        wav, feat = (
            batch[0].to(self.device), 
            batch[1].to(self.device)
        )

        feat_hat = self.model(wav)
        feat_hat = feat_hat.transpose(2, 1)
        feat_hat = LibriTTSRDataset.pad_feat(feat_hat, device=self.device)
        feat_hat = feat_hat.transpose(2, 1)

        p_hat = self.discriminator(feat_hat)

        gen_loss = self.gen_adv_loss(p_hat)

        feat_hat = feat_hat.transpose(2, 1)
        ema_loss = self.ema_loss(feat_hat, feat)

        loss = gen_loss + 45 * ema_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss
    
    def disc_step(self, batch):
        wav, feat = (
            batch[0].to(self.device), 
            batch[1].to(self.device)
        )
        feat = feat.transpose(2, 1)

        with torch.no_grad():
            feat_hat = self.model(wav)
            feat_hat = feat_hat.transpose(2, 1)
            feat_hat = LibriTTSRDataset.pad_feat(feat_hat, device=self.device)
            feat_hat = feat_hat.transpose(2, 1)
        
        p = self.discriminator(feat)
        p_hat = self.discriminator(feat_hat)

        real_loss, fake_loss = self.disc_adv_loss(p, p_hat)
        loss = real_loss + fake_loss

        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        self.disc_scheduler.step()

        return loss
    
    def save_checkpoint(self, val_loss: float):
        model_dict = {
            "model_state": self.model.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "disc_optimizer_state": self.disc_optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "disc_scheduler_state": self.disc_scheduler.state_dict(),
        }
        ckpt_path = Configs.generate_checkpoint_dir(self.config, val_loss)
        torch.save(model_dict, self.config_dir.parent / ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    config_dir = pathlib.Path(args.config)
    config = Configs.load_config(config_dir)

    train_dataset, val_dataset, test_dataset = Configs.load_dataset(config)
    train_dataloader = Configs.load_dataloader(config, train_dataset)
    val_dataloader = Configs.load_dataloader(config, val_dataset)

    model = Configs.load_model(config=config)
    discriminator = Configs.load_discriminator(config=config)

    criterions = {
        "l1": torch.nn.L1Loss,
        "disc_adv": DiscriminatorAdversarialLoss,
        "gen_adv": GeneratorAdversarialLoss
    }
    ema_loss = Configs.load_criterion(config, criterions)
    gen_adv_loss = Configs.load_criterion(config, criterions, "gen_adv")
    disc_adv_loss = Configs.load_criterion(config, criterions, "disc_adv")

    optimizer = Configs.load_optimizer(config, model)
    scheduler = Configs.load_scheduler(config, optimizer)

    disc_optimizer = Configs.load_optimizer(config, discriminator)
    disc_scheduler = Configs.load_scheduler(config, disc_optimizer)

    train_steps = len(train_dataloader)
    val_steps = len(val_dataloader)

    trainer = Trainer(
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        discriminator=discriminator,
        ema_loss=ema_loss,
        gen_adv_loss=gen_adv_loss,
        disc_adv_loss=disc_adv_loss,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        scheduler=scheduler,
        disc_scheduler=disc_scheduler,
        checkpoint_interval=5000,
        config=config,
        config_dir=config_dir,
        device=0
    )

    trainer.train()
