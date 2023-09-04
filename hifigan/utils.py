import torch
import matplotlib
import glob
import os

matplotlib.use("Agg")
import matplotlib.pylab as plt


def get_padding(k, d):
    return int((k * d - d) / 2)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def save_checkpoint(
    checkpoint_dir,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    step,
    loss,
    best,
    logger,
):
    state = {
        "generator": {
            "model": generator.state_dict(),
            "optimizer": optimizer_generator.state_dict(),
            "scheduler": scheduler_generator.state_dict(),
        },
        "discriminator": {
            "model": discriminator.state_dict(),
            "optimizer": optimizer_discriminator.state_dict(),
            "scheduler": scheduler_discriminator.state_dict(),
        },
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_path = checkpoint_dir / f"hifigan-training-model-{step}.pt"
    torch.save(state, checkpoint_path)
    
    if best:
        model_path = checkpoint_dir / "hifigan-model.pt"
        torch.save(state["generator"]["model"], model_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")
    old_model = oldest_checkpoint_path(checkpoint_dir, logger)
    if os.path.exists(old_model):
        os.remove(old_model)
        print(f"Removed {old_model}")


def load_checkpoint(
    load_path,
    generator,
    discriminator,
    optimizer_generator,
    optimizer_discriminator,
    scheduler_generator,
    scheduler_discriminator,
    rank,
    logger,
    finetune=False,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    generator.load_state_dict(checkpoint["generator"]["model"])
    discriminator.load_state_dict(checkpoint["discriminator"]["model"])
    if not finetune:
        optimizer_generator.load_state_dict(checkpoint["generator"]["optimizer"])
        scheduler_generator.load_state_dict(checkpoint["generator"]["scheduler"])
        optimizer_discriminator.load_state_dict(
            checkpoint["discriminator"]["optimizer"]
        )
        scheduler_discriminator.load_state_dict(
            checkpoint["discriminator"]["scheduler"]
        )
    return checkpoint["step"], checkpoint["loss"]

def extract_digits(f):
    digits = "".join(filter(str.isdigit, f))
    return int(digits) if digits else -1


def latest_checkpoint_path(dir_path, regex="hifigan-training-model-[0-9]*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    print(f_list)
    f_list.sort(key=lambda f: extract_digits(f))
    x = f_list[-1]
    print(f"latest_checkpoint_path:{x}")
    return x


def oldest_checkpoint_path(dir_path, logger, regex="hifigan-training-model-[0-9]*.pt", preserved=2):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: extract_digits(f))
    if len(f_list) > preserved:
        x = f_list[0]
        logger.info(f"oldest_checkpoint_path:{x}")
        return x
    return ""
