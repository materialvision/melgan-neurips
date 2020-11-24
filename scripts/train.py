#!/usr/bin/env python
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader

from mel2wav.dataset import AudioDataset
from mel2wav.modules import Audio2Mel, Discriminator, Generator
from mel2wav.utils import save_sample
from util import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--load_from_run_id", default=None)
    parser.add_argument("--resume_run_id", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--pad_mode", type=str, default="reflect", choices=["reflect", "replicate"]
    )

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--sampling_rate", type=int, default=22050)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args


def main():

    seed_everything(7)

    args = parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    entity = "demiurge"
    project = "melgan"
    load_from_run_id = args.load_from_run_id
    resume_run_id = args.resume_run_id
    restore_run_id = load_from_run_id or resume_run_id
    load_initial_weights = bool(restore_run_id)
    sampling_rate = args.sampling_rate

    if load_from_run_id and resume_run_id:
        raise RuntimeError("Specify either --load_from_id or --resume_run_id.")

    if resume_run_id:
        print(f"Resuming run ID {resume_run_id}.")
    elif load_from_run_id:
        print(f"Starting new run with initial weights from run ID {load_from_run_id}.")
    else:
        print("Starting new run from scratch.")

    wandb.init(
        entity=entity,
        project=project,
        id=resume_run_id,
        config=args,
        resume=True if resume_run_id else False,
        save_code=True,
        dir=args.save_path,
    )

    print("run id: " + str(wandb.run.id))
    print("run name: " + str(wandb.run.name))
    
    root = Path(wandb.run.dir)
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).cuda()
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).cuda()
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels, pad_mode=args.pad_mode).cuda()

    for model in [netG, netD, fft]:
        wandb.watch(model)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    if load_initial_weights:

        for obj, filename in [
            (netG, "netG.pt"),
            (optG, "optG.pt"),
            (netD, "netD.pt"),
            (optD, "optD.pt"),
        ]:
            run_path = f"{entity}/{project}/{restore_run_id}"
            print(f"Restoring {filename} from run path {run_path}")
            restored_file = wandb.restore(filename, run_path=run_path)
            obj.load_state_dict(torch.load(restored_file.name))

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=sampling_rate
    )
    test_set = AudioDataset(
        Path(args.data_path) / "test_files.txt",
        sampling_rate * 4,
        sampling_rate=sampling_rate,
        augment=False,
    )
    wandb.save(str(Path(args.data_path) / "train_files.txt"))
    wandb.save(str(Path(args.data_path) / "test_files.txt"))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1)

    if len(train_loader) == 0:
        raise RuntimeError("Train dataset is empty.")

    if len(test_loader) == 0:
        raise RuntimeError("Test dataset is empty.")

    # Getting initial run steps and epoch

    if load_from_run_id:
        api = wandb.Api()
        previous_run = api.run(f"{entity}/{project}/{restore_run_id}")
        steps = previous_run.lastHistoryStep
    else:
        steps = wandb.run.step

    start_epoch = steps // len(train_loader)
    print(f"Starting with epoch {start_epoch} and step {steps}.")

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    samples = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t.cuda()
        s_t = fft(x_t).detach()

        test_voc.append(s_t.cuda())
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), sampling_rate, audio)
        samples.append(wandb.Audio(audio, caption=f"sample {i}", sample_rate=sampling_rate))

        if i == args.n_test_samples - 1:
            break

    if not resume_run_id:
        wandb.log({"audio/original": samples}, step=0)
    else:
        print("We are resuming, skipping logging of original audio.")

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000

    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t.cuda()
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t.cuda())

            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()

            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.cuda().detach())
            D_real = netD(x_t.cuda())

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.cuda())

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()
            optG.step()

            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            wandb.log(
                {
                    "loss/discriminator": costs[-1][0],
                    "loss/generator": costs[-1][1],
                    "loss/feature_matching": costs[-1][2],
                    "loss/mel_reconstruction": costs[-1][3],
                },
                step=steps,
            )
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    samples = []
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), sampling_rate, pred_audio)
                        samples.append(
                            wandb.Audio(
                                pred_audio,
                                caption=f"sample {i}",
                                sample_rate=sampling_rate,
                            )
                        )
                    wandb.log(
                        {
                            "audio/generated": samples,
                            "epoch": epoch,
                        },
                        step=steps,
                    )

                print("Saving models ...")
                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")
                wandb.save(str(root / "netG.pt"))
                wandb.save(str(root / "optG.pt"))

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")
                wandb.save(str(root / "netD.pt"))
                wandb.save(str(root / "optD.pt"))

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")
                    wandb.save(str(root / "best_netD.pt"))
                    wandb.save(str(root / "best_netG.pt"))

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()

if __name__ == "__main__":
    main()