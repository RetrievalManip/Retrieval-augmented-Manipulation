from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAM_DIR = PROJECT_ROOT / "visions" / "ram"

for path in (PROJECT_ROOT, RAM_DIR / "lib", RAM_DIR / "data", Path(__file__).resolve().parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RAM pose model on preprocessed BOP-style training data."
    )
    parser.add_argument("--dataset", type=str, default="bop", choices=["bop"])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "BOP",
        help="Directory containing RAM training lists, template pkl files, cad metadata, and rendered samples.",
    )
    parser.add_argument("--foundation-model", type=str, default="dinov2-b14", choices=["dinov2-b14"])
    parser.add_argument("--feat-layers", type=int, nargs="+", default=[7, 9, 11])
    parser.add_argument("--feat-type", type=str, default="k", choices=["q", "k", "v"])
    parser.add_argument("--n-pts", type=int, default=1024)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--template-views", type=int, default=128)
    parser.add_argument(
        "--pretrained-dino-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "visions" / "ram" / "checkpoints" / "dinov2_vitb14_pretrain_torch1.pth",
        help="DINOv2-B/14 backbone checkpoint used to initialize the RAM foundation model.",
    )
    parser.add_argument(
        "--result-root-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ram_training",
    )
    parser.add_argument("--exp-name", type=str, default="ram-dinov2-b14")
    parser.add_argument("--resume-epoch", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay-epoch", type=int, nargs="+", default=[0, 6, 12, 18, 24])
    parser.add_argument("--decay-rate", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125, 0.0625])
    parser.add_argument("--pose-nce-tau", type=float, default=0.05)
    parser.add_argument("--match-loss-weight", type=float, default=0.001)
    parser.add_argument("--nocs-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-log-interval", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="RAMNet-train")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--disable-template-refresh", action="store_true")
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def resolve_result_dir(args: argparse.Namespace) -> Path:
    result_dir = args.result_root_dir / args.dataset / args.exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_chamfer_distance():
    try:
        from pytorch3d.loss import chamfer_distance

        return chamfer_distance
    except ImportError:
        import torch

        def fallback_chamfer_distance(
            x,
            y,
            batch_reduction: str = "mean",
            weights=None,
            single_directional: bool = False,
            **_: Any,
        ):
            distances = torch.cdist(x, y)
            loss = distances.min(dim=2).values.mean(dim=1)
            if not single_directional:
                loss = loss + distances.min(dim=1).values.mean(dim=1)
            if weights is not None:
                weights = weights.reshape(-1).to(device=loss.device, dtype=loss.dtype)
                loss = loss * weights
            if batch_reduction == "sum":
                return loss.sum(), None
            if batch_reduction is None:
                return loss, None
            return loss.mean(), None

        return fallback_chamfer_distance


class MetricLogger:
    def __init__(self, args: argparse.Namespace):
        self._wandb = None
        if args.disable_wandb:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is required unless --disable-wandb is set.") from exc

        self._wandb = wandb
        self._wandb.init(project=args.wandb_project, name=args.exp_name)

    def watch(self, model: Any) -> None:
        if self._wandb is not None:
            self._wandb.watch(model)

    def log(self, metrics: dict[str, Any]) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics)


def build_model(args: argparse.Namespace, device):
    import torch.nn as nn
    from network import RAMNet

    require_path(args.pretrained_dino_checkpoint, "DINOv2-B/14 pretrain checkpoint")

    model = RAMNet(
        foundation_model=args.foundation_model,
        feat_layers=args.feat_layers,
        feat_type=args.feat_type,
        img_size=args.img_size,
    )
    model.load_pretrain_checkpoint(str(args.pretrained_dino_checkpoint))
    model.to(device)

    if device.type == "cuda" and not args.no_data_parallel:
        model = nn.DataParallel(model)

    return model


def trainable_parameters(model):
    for name, parameter in model.named_parameters():
        if "foundation_model" in name:
            parameter.requires_grad_(False)
            continue
        yield parameter


def build_datasets(args: argparse.Namespace):
    from bop_dataset import RAMBOPDataset

    require_path(args.data_dir, "RAM training data directory")
    return (
        RAMBOPDataset(
            args.dataset,
            "train",
            str(args.data_dir),
            args.n_pts,
            args.img_size,
            feat_layers=args.feat_layers,
            feat_type=args.feat_type,
            foundation_model=args.foundation_model,
            template_views=args.template_views,
        ),
        RAMBOPDataset(
            args.dataset,
            "test",
            str(args.data_dir),
            args.n_pts,
            args.img_size,
            feat_layers=args.feat_layers,
            feat_type=args.feat_type,
            foundation_model=args.foundation_model,
            template_views=args.template_views,
        ),
    )


def move_batch_to_device(data: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in data.items()
    }


def make_subset_loader(dataset, sample_count: int, batch_size: int, num_workers: int, torch_module):
    sample_count = min(sample_count, dataset.length)
    indices = random.sample(list(range(dataset.length)), sample_count)
    sampler = torch_module.utils.data.sampler.SubsetRandomSampler(indices)
    return torch_module.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def optimizer_for_epoch(args: argparse.Namespace, model, epoch: int, torch_module):
    rate_id = 0
    for idx, decay_epoch in enumerate(args.decay_epoch):
        if epoch > decay_epoch:
            rate_id = idx
    current_lr = args.lr * args.decay_rate[min(rate_id, len(args.decay_rate) - 1)]
    return torch_module.optim.Adam(list(trainable_parameters(model)), lr=current_lr), current_lr


def train_one_epoch(
    args: argparse.Namespace,
    model,
    dataset,
    optimizer,
    losses,
    chamfer_distance,
    logger: MetricLogger,
    device,
    torch_module,
    start_time: float,
) -> None:
    train_size = args.train_steps * args.batch_size
    train_loader = make_subset_loader(dataset, train_size, args.batch_size, args.num_workers, torch_module)

    pose_criterion, nocs_criterion, match_criterion = losses
    model.train()
    print("########################## start training ##########################")

    for step, raw_data in enumerate(train_loader, 1):
        data = move_batch_to_device(raw_data, device)

        prediction = model(
            data["query_rgb"],
            data["query_choose"],
            data["sampled_template_feat"],
            data["sampled_position_encoding"],
            data["select_template_feat"],
            data["select_template_nocs"],
            data["query_partial_shape"],
            data["query_shape_in_template_view"],
            data["sampled_template_partial_shape"],
            data["select_template_partial_shape"],
            data["select_template_shape_in_obj_frame"],
            data["hard_sampled_template_feat"],
            data["hard_sampled_position_encoding"],
            data["hard_sampled_template_partial_shape"],
            data["template_complete_nocs"],
        )

        view_loss = pose_criterion(
            prediction["query_view_feat"],
            prediction["template_view_feat"],
            data["view_error"],
            data["top_1_view_choose"],
        )
        hard_view_loss = pose_criterion(
            prediction["query_view_feat"],
            prediction["local_template_view_feat"],
            data["hard_view_error"],
            data["top_1_view_choose"],
        )
        match_loss = match_criterion(prediction["match_matrix"])
        recons_loss, _ = chamfer_distance(
            prediction["recons_nocs"],
            data["query_complete_nocs"],
            single_directional=True,
        )

        nocs_loss1 = nocs_criterion(
            data["query_nocs"],
            prediction["nocs_pred"],
            1.0 - data["is_symmetry"],
        )
        nocs_loss2, _ = chamfer_distance(
            prediction["nocs_pred"],
            data["query_complete_nocs"],
            batch_reduction="sum",
            weights=data["is_symmetry"],
            single_directional=True,
        )
        nocs_loss = (nocs_loss1 + nocs_loss2) / args.batch_size
        total_loss = view_loss + hard_view_loss + match_loss + recons_loss + nocs_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.log(
            {
                "loss": total_loss,
                "view_loss": view_loss,
                "hard_view_loss": hard_view_loss,
                "nocs_loss": nocs_loss,
                "match_loss": match_loss,
                "recons_loss": recons_loss,
            }
        )

        if step % args.log_interval == 0:
            elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
            print(
                f"Time {elapsed}, Batch {step} Loss:{total_loss.item():f}, "
                f"view Loss:{view_loss.item():f}, hard view Loss:{hard_view_loss.item():f}, "
                f"nocs Loss:{nocs_loss.item():f}, match_loss:{match_loss.item():f}, "
                f"recons_loss:{recons_loss.item():f}"
            )

    print("########################## training finished ##########################")


def validate(
    args: argparse.Namespace,
    model,
    dataset,
    losses,
    chamfer_distance,
    logger: MetricLogger,
    device,
    torch_module,
) -> None:
    import numpy as np

    pose_criterion, nocs_criterion, match_criterion = losses
    val_size = min(args.val_size, dataset.length)
    val_loader = make_subset_loader(dataset, val_size, 1, args.num_workers, torch_module)

    model.eval()
    total_num = 0
    correct_num = 0
    val_nocs_loss = 0.0
    val_recons_loss = 0.0
    val_view_loss = 0.0

    print("########################## start validation ##########################")
    for step, raw_data in enumerate(val_loader, 1):
        data = move_batch_to_device(raw_data, device)

        with torch_module.no_grad():
            prediction = model(
                data["query_rgb"],
                data["query_choose"],
                data["sampled_template_feat"],
                data["sampled_position_encoding"],
                data["select_template_feat"],
                data["select_template_nocs"],
                data["query_partial_shape"],
                data["query_shape_in_template_view"],
                data["sampled_template_partial_shape"],
                data["select_template_partial_shape"],
                data["select_template_shape_in_obj_frame"],
                None,
                None,
                None,
                data["template_complete_nocs"],
            )

        view_loss = pose_criterion(
            prediction["query_view_feat"],
            prediction["template_view_feat"],
            data["view_error"],
            data["top_1_view_choose"],
        )
        match_loss = match_criterion(prediction["match_matrix"])
        recons_loss, _ = chamfer_distance(prediction["recons_nocs"], data["query_complete_nocs"])
        nocs_loss1 = nocs_criterion(
            data["query_nocs"],
            prediction["nocs_pred"],
            1.0 - data["is_symmetry"],
        )
        nocs_loss2, _ = chamfer_distance(
            prediction["nocs_pred"],
            data["query_complete_nocs"],
            batch_reduction="sum",
            weights=data["is_symmetry"],
            single_directional=True,
        )
        nocs_loss = nocs_loss1 + nocs_loss2
        total_loss = view_loss + match_loss + recons_loss + nocs_loss

        val_view_loss += view_loss.item()
        val_recons_loss += recons_loss.item()
        val_nocs_loss += nocs_loss.item()

        pred_view = np.argmax(prediction["view_pred"].detach().cpu().numpy(), axis=1)
        gt_view = np.argmax(data["sampled_template_view_score"].detach().cpu().numpy(), axis=1)
        total_num += 1
        if pred_view[0] == gt_view[0]:
            correct_num += 1

        if step % args.val_log_interval == 0:
            print(
                f"Batch {step} Loss:{total_loss.item():f}, view Loss:{view_loss.item():f}, "
                f"nocs Loss:{nocs_loss.item():f}, match_loss:{match_loss.item():f}, "
                f"recons_loss:{recons_loss.item():f}"
            )

    val_accuracy = float(correct_num) / float(total_num)
    print(f"Accuracy:{val_accuracy:f}")
    logger.log(
        {
            "val_accuracy": val_accuracy,
            "val_view_loss": val_view_loss / val_size,
            "val_recons_loss": val_recons_loss / val_size,
            "val_nocs_loss": val_nocs_loss / val_size,
        }
    )
    print("########################## validation finished ##########################")


def resolve_resume_path(args: argparse.Namespace, result_dir: Path) -> Path | None:
    if args.resume_checkpoint is not None:
        return args.resume_checkpoint
    if args.resume_epoch is not None:
        return result_dir / f"model_{args.resume_epoch:02d}.pth"
    return None


def state_dict_for_ram_inference(model) -> dict[str, Any]:
    if hasattr(model, "module"):
        return model.state_dict()
    return {f"module.{key}": value for key, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    if len(args.decay_epoch) != len(args.decay_rate):
        raise ValueError("--decay-epoch and --decay-rate must have the same length.")

    import torch

    from losses import MatchLoss, NOCSLoss, PoseNCE

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu for CPU debugging.")

    result_dir = resolve_result_dir(args)
    model = build_model(args, device)
    resume_path = resolve_resume_path(args, result_dir)
    if resume_path is not None:
        require_path(resume_path, "resume checkpoint")
        model.load_state_dict(torch.load(resume_path, map_location=device))

    logger = MetricLogger(args)
    logger.watch(model)

    train_dataset, val_dataset = build_datasets(args)
    losses = (
        PoseNCE(args.pose_nce_tau),
        NOCSLoss(args.nocs_loss_weight),
        MatchLoss(args.match_loss_weight),
    )
    chamfer_distance = load_chamfer_distance()
    start_time = time.time()

    for epoch in range(args.start_epoch, args.max_epoch + 1):
        optimizer, current_lr = optimizer_for_epoch(args, model, epoch, torch)
        logger.log({"learning_rate": current_lr, "epoch": epoch})
        print(f"########################## epoch {epoch} lr {current_lr:g} ##########################")

        train_one_epoch(
            args,
            model,
            train_dataset,
            optimizer,
            losses,
            chamfer_distance,
            logger,
            device,
            torch,
            start_time,
        )
        validate(args, model, val_dataset, losses, chamfer_distance, logger, device, torch)

        checkpoint_path = result_dir / f"model_{epoch:02d}.pth"
        torch.save(state_dict_for_ram_inference(model), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if not args.disable_template_refresh:
            train_dataset.change_template()


if __name__ == "__main__":
    main()
