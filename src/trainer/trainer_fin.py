import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
import json
import wandb

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from src.datasets.dataset import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from src.model.model_logamp import (
    Generator,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    feature_loss,
    ls_discriminator_loss,
    ls_generator_loss
)
from src.utils.utils import (
    AttrDict,
    build_env,
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

torch.backends.cudnn.benchmark = True


def train(h):

    wandb.init(
        project="audio_vocoder",  
        config=h,                    
        name=f"experiment",  
        sync_tensorboard=True,
    )

    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:{:d}".format(0))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    print(generator)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, "g_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    training_filelist, validation_filelist = get_dataset_filelist(
        h.input_training_wav_list, h.input_validation_wav_list, h.raw_wavfile_path
    )

    trainset = Dataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        n_cache_reuse=0,
        shuffle=True,
        inv_mel=True,
        use_pghi=True,
        device=device,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validset = Dataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        False,
        shuffle=False,
        n_cache_reuse=0,
        device=device,
        inv_mel=True,
        use_pghi=True,
    )
    
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    mrd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(train_loader):
            start_b = time.time()
            x, logamp, pha, rea, imag, y, meloss, inv_mel, pghid = map(
                lambda x: x.to(device, non_blocking=True), batch
            )
            # y = y.unsqueeze(1)
            y_g = generator(x, pghi=pghid)

            y_min = np.min([y_g.shape[-1], y.shape[-1]])
            y_g, y = y_g[..., :y_min], y[..., :y_min]
 
            y_g_mel = mel_spectrogram(
                y_g,
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.meloss,
            )

            optim_d.zero_grad()

            y = y.unsqueeze(1)
            y_g = y_g.unsqueeze(1)

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())

            loss_disc_f, losses_disc_f_r, losses_disc_f_g = ls_discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())

            loss_disc_s, losses_disc_s_r, losses_disc_s_g = ls_discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

            L_D = loss_disc_s + loss_disc_f

            L_D.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            _, _, rea_g_final, imag_g_final = amp_pha_specturm(
                y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size
            )

            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = ls_generator_loss(y_df_g)
            loss_gen_s, losses_gen_s = ls_generator_loss(y_ds_g)
            L_GAN_G = loss_gen_s + loss_gen_f
            L_FM = loss_fm_s + loss_fm_f
            L_Mel = F.l1_loss(meloss, y_g_mel)
            L_G = L_GAN_G + L_FM + 45 * L_Mel

            L_G.backward()
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    Mel_error = F.l1_loss(x, y_g_mel).item()

                print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, s/b : {:4.3f}".format(
                        steps,
                        L_G,
                        Mel_error,
                        time.time() - start_b,
                    )
                )

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"generator": generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": mpd.state_dict(),
                        "mrd": mrd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )

            if steps % h.summary_interval == 0:
                wandb.log({
                            "Training/Generator_Total_Loss": L_G.item(),
                            "Training/Mel_Spectrogram_Loss": Mel_error,
                        }, step=steps)
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)


            # Validation
            if steps % h.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()

                val_Mel_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):

                        x, logamp, pha, rea, imag, y, meloss, inv_mel, pghid = map(
                            lambda x: x.to(device, non_blocking=True), batch
                        )

                        y_g = generator(
                                x, pghi=pghid
                            )
                        
                        y_min = np.min([y_g.shape[-1], y.shape[-1]])
                        y_g, y = y_g[..., :y_min], y[..., :y_min]
            
                        y_g_mel = mel_spectrogram(
                            y_g,
                            h.n_fft,
                            h.num_mels,
                            h.sampling_rate,
                            h.hop_size,
                            h.win_size,
                            h.fmin,
                            h.meloss,
                        )

                        y = y.unsqueeze(1)
                        y_g = y_g.unsqueeze(1)

                        val_Mel_err_tot += F.l1_loss(meloss, y_g_mel).item()

                        if j <= 4:

                            if steps == 0:
                                sw.add_audio(
                                    "gt/y_{}".format(j), y[0], steps, h.sampling_rate
                                )
                                sw.add_figure(
                                    "gt/y_spec_{}".format(j),
                                    plot_spectrogram(x[0].cpu()),
                                    steps,
                                )

                            sw.add_audio(
                                "generated/y_g_{}".format(j),
                                y_g[0],
                                steps,
                                h.sampling_rate,
                            )
                            y_g_spec = mel_spectrogram(
                                y_g.squeeze(1),
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                "generated/y_g_spec_{}".format(j),
                                plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()),
                                steps,
                            )
                            # print('y', y.shape)
                            wandb.log({
                                        f"Audio/gt_y_{j}": wandb.Audio(y.squeeze(1)[0].cpu().numpy(), sample_rate=h.sampling_rate),
                                        f"Audio/gen_y_g_{j}": wandb.Audio(y_g.squeeze(1)[0].cpu().numpy(), sample_rate=h.sampling_rate),
                                        f"Spectrogram/gt_y_spec_{j}": wandb.Image(plot_spectrogram(x[0].cpu())),
                                        f"Spectrogram/gen_y_g_spec_{j}": wandb.Image(plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()))
                                    }, step=steps)

                    val_Mel_err = val_Mel_err_tot / (j + 1)

                    wandb.log({
                              "Validation/Mel_Spectrogram_loss": val_Mel_err,
                          }, step=steps)
                    
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)

                generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch + 1, int(time.time() - start)
            )
        )

    wandb.finish()


def main():
    print("Initializing Training Process..")

    config_file = "src/configs/model_fin.json"

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, "config_fin.json", h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == "__main__":
    main()
