import argparse
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from models.mot_detr import MOTDETR


def train_net(
    batch_size,
    hidden_dim,
    lr,
    weight_decay,
    max_epochs,
    nheads,
    num_queries,
    num_encoder_layers,
    num_decoder_layers,
    scheduler_steps,
    num_classes,
    num_track_classes,
):
    trainer = pl.Trainer(
        default_root_dir="/mnt/data_ssd/code/repos/3d-trackformer/",
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_ap", every_n_epochs=1, save_last=True),
            LearningRateMonitor("step"),
        ],
        precision="bf16-mixed",
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    dataset_6c_train = "Load your train dataset here"
    dataset_6c_val = "Load your validation dataset here"

    train_loader = data.DataLoader(dataset_6c_train, num_workers=16, shuffle=True, batch_size=batch_size)
    val_loader = data.DataLoader(dataset_6c_val, num_workers=4, shuffle=False, batch_size=1)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = "lightning_logs/version_0/checkpoints/last.ckpt"
    resume = False
    if resume:
        print(f"Resuming from pretrained model at {pretrained_filename}, loading...")
        model = MOTDETR.load_from_checkpoint(
            pretrained_filename,
            lr=1.25e-5,
            max_epochs=30,
            batch_size=batch_size,
            scheduler_steps=[8000, 26000, 36000, 70000],
        )  # Automatically loads the model with the saved hyperparameters
        # trainer.validate(model, val_loader)
        trainer.fit(model, train_loader, val_loader)
    else:
        model = MOTDETR(
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            num_classes=num_classes,
            num_track_classes=num_track_classes,
            nheads=nheads,
            num_queries=num_queries,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_size=batch_size,
        )
        trainer.fit(model, train_loader, val_loader)


def main(args):
    train_net(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        nheads=args.nheads,
        num_queries=args.num_queries,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        scheduler_steps=args.scheduler_steps,
        num_classes=args.num_classes,
        num_track_classes=args.num_track_classes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=30)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--scheduler_steps", type=int, nargs="+", default=[30000, 50000, 65000, 80000])
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_track_classes", type=int, default=1)

    args = parser.parse_args()

    main(args)
