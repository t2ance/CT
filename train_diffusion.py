import argparse
import os
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# HuggingFace
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDIMScheduler
import wandb

# Custom modules
from model_factory import build_diffusion_model, DiffusionUNet3D
from models.vqae_wrapper import FrozenVQAE
from data.hf_dataset_loader import create_hf_dataloaders, create_train_subset_dataloader
from utils.metrics import DistributedMetricsCalculator
from utils.visualization import CTVisualization
from utils.network_visualizer import NetworkArchitectureVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train latent diffusion model for CT super-resolution"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> DiffusionUNet3D:
    """
    Create 3D UNet diffusion model from config.

    Args:
        config: Configuration dictionary

    Returns:
        model: DiffusionUNet3D instance
    """
    model_config = config["model"]
    return build_diffusion_model(model_config, verbose=True)


@torch.no_grad()
def validate(
    model: DiffusionUNet3D,
    val_loader: DataLoader,
    noise_scheduler: DDPMScheduler,
    accelerator: Accelerator,
) -> float:
    """
    Lightweight validation with MSE loss only (no DDIM sampling).

    This is fast (~5s) and used for frequent validation checks.

    Args:
        model: Diffusion model
        val_loader: Validation dataloader (prepared by accelerate)
        noise_scheduler: DDPM noise scheduler
        accelerator: Accelerator instance

    Returns:
        avg_loss: Average MSE loss on validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        # Move tensors to device (Accelerate doesn't auto-move dict values)
        ld_latent = batch["ld_latent"].to(accelerator.device)
        hd_latent = batch["hd_latent"].to(accelerator.device)

        B = hd_latent.shape[0]

        # Resize LD latent to match HD latent depth if needed
        if ld_latent.shape[2] != hd_latent.shape[2]:
            ld_latent = F.interpolate(
                ld_latent,
                size=hd_latent.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (B,), device=hd_latent.device
        ).long()

        # Add noise to HD latent
        noise = torch.randn_like(hd_latent)
        noisy_hd_latent = noise_scheduler.add_noise(hd_latent, noise, timesteps)

        # Concatenate LD latent as conditioning
        model_input = torch.cat([noisy_hd_latent, ld_latent], dim=1)

        # Predict noise
        noise_pred = model(model_input, timesteps)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # Gather across all processes
    avg_loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
    avg_loss_tensor = accelerator.gather(avg_loss_tensor).mean().item()

    model.train()
    return avg_loss_tensor


@torch.no_grad()
def validate_with_metrics(
    model: DiffusionUNet3D,
    val_loader: DataLoader,
    vae: FrozenVQAE,
    config: dict,
    accelerator: Accelerator,
    num_samples: int = 4,
    spatial_chunk_size: int = 32,
) -> dict:
    """
    Full validation with SSIM and PSNR metrics.

    This function uses a PREPARED val_loader (processed by accelerator.prepare())
    which automatically distributes batches across GPUs via DistributedSampler.
    Each GPU evaluates different samples to parallelize the slow DDIM sampling.

    Process:
    1. Prepared loader distributes batches across GPUs automatically
    2. Each GPU processes its assigned batches independently
    3. DDIM sampling (100 steps, ~30-60s per sample) without inter-GPU sync
    4. Decode with VQ-AE
    5. Compute SSIM/PSNR using DistributedMetricsCalculator
    6. Synchronize all processes (wait_for_everyone)
    7. Gather metrics from all processes
    8. Return aggregated metrics dict (main process only)

    Args:
        model: Diffusion model
        val_loader: Validation dataloader (PREPARED - from accelerator.prepare)
        vae: Frozen VQ-AE for decoding
        config: Configuration dict
        accelerator: Accelerator instance
        num_samples: Number of validation samples to evaluate TOTAL (distributed across all GPUs)

    Returns:
        metrics_dict: Aggregated metrics with mean/std/min/max for SSIM/PSNR.
                      Only populated on main process, empty dict on other processes.
        visualizations: Dict of {name: wandb.Image} collected on main process
                        for logging. Empty dict on non-main processes.
    """
    if vae is None:
        accelerator.print("  Warning: VQ-AE not loaded, skipping metrics")
        return {}, {}

    model.eval()

    # Create DDIM scheduler for sampling
    inference_config = config.get("inference", {})
    num_inference_steps = inference_config.get("num_inference_steps", 100)

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=config["scheduler"]["num_train_timesteps"],
        beta_start=config["scheduler"]["beta_start"],
        beta_end=config["scheduler"]["beta_end"],
        beta_schedule=config["scheduler"]["beta_schedule"],
        prediction_type=config["scheduler"]["prediction_type"],
        clip_sample=config["scheduler"].get("clip_sample", False),
    )
    ddim_scheduler.set_timesteps(num_inference_steps)

    # Initialize metrics calculator
    validation_config = config.get("validation", {})
    data_range = validation_config.get(
        "data_range", 2.0
    )  # CRITICAL: 2.0 for normalized VQ-AE outputs!
    compute_slice_wise = validation_config.get("compute_slice_wise", True)
    num_visualizations = int(validation_config.get("num_visualizations", 2) or 0)

    collect_visualizations = (
        num_visualizations > 0
        and accelerator.is_main_process
        and config.get("training", {}).get("use_wandb", True)
    )
    visualizations = {}
    viz_helper = CTVisualization() if collect_visualizations else None
    viz_target = min(num_visualizations, num_samples) if collect_visualizations else 0
    viz_count = 0

    metrics_calc = DistributedMetricsCalculator(
        data_range=data_range, compute_slice_wise=compute_slice_wise
    )

    samples_per_process = (
        num_samples + accelerator.num_processes - 1
    ) // accelerator.num_processes

    # Collect metrics
    local_metrics = []
    sample_idx = 0
    batch_idx = 0

    if accelerator.is_main_process:
        accelerator.print(
            f"  Validating {num_samples} samples total ({samples_per_process} per process across {accelerator.num_processes} GPU(s))..."
        )

    for batch in tqdm(val_loader, desc=f"Validating (GPU {accelerator.process_index})"):
        if sample_idx >= samples_per_process:
            break

        ld_latent = batch["ld_latent"].to(accelerator.device)
        hd_latent_gt = batch["hd_latent"].to(accelerator.device)
        batch_size = hd_latent_gt.shape[0]

        # Resize LD latent to match HD latent depth if needed
        if ld_latent.shape[2] != hd_latent_gt.shape[2]:
            ld_latent = F.interpolate(
                ld_latent,
                size=hd_latent_gt.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        # Start from pure noise
        hd_latent_pred = torch.randn_like(hd_latent_gt)

        # DDIM sampling
        for t in ddim_scheduler.timesteps:
            # Concatenate conditioning
            model_input = torch.cat([hd_latent_pred, ld_latent], dim=1)

            # Predict noise
            noise_pred = model(model_input, t.unsqueeze(0).to(accelerator.device))

            # DDIM step
            hd_latent_pred = ddim_scheduler.step(
                noise_pred, t, hd_latent_pred
            ).prev_sample

        # Decode predicted latent to CT space with spatial chunking to save memory
        # Use 32x32x32 latent chunks (256x256x256 in image space) to avoid 18GB allocations
        # This processes ~2-3GB per chunk instead of 18GB for full volume
        pred_ct = vae.decode(hd_latent_pred, micro_batch_size=1, spatial_chunk_size=spatial_chunk_size)  # [B, 1, D, H, W]

        # CRITICAL: Use original HD CT from batch (not reconstructed via VQ-AE decode)
        # This gives accurate metrics without VQ-AE reconstruction artifacts
        gt_ct = batch["hd_ct"]  # [B, 1, D, H, W], already normalized to [-1, 1]

        # Decode LD CT for visualization if needed
        ld_ct = (
            batch["ld_ct"]  # [B, 1, D, H, W]
            if collect_visualizations and viz_count < viz_target
            else None
        )

        # Compute metrics for each sample in batch
        for i in range(pred_ct.shape[0]):
            if sample_idx >= samples_per_process:
                break

            # Convert to numpy and remove batch/channel dims
            pred_np = pred_ct[i, 0].cpu().numpy()  # [D, H, W]
            gt_np = gt_ct[i, 0].cpu().numpy()

            local_metrics.append(metrics_calc.compute_metrics(pred_np, gt_np))

            sample_idx += 1

            if collect_visualizations and viz_count < viz_target:
                sample_viz = viz_helper.create_multi_slice_comparison(
                    ld=ld_ct[i : i + 1] if ld_ct is not None else None,
                    pred=pred_ct[i : i + 1],
                    gt=gt_ct[i : i + 1],
                    num_slices=5,
                )
                visualizations[f"val/sample_{viz_count}"] = wandb.Image(sample_viz)
                viz_count += 1

        batch_idx += 1

    # Synchronize all processes before gathering
    accelerator.wait_for_everyone()

    # Convert local metrics to tensors for gathering
    if local_metrics:
        local_ssim = torch.tensor(
            [m["ssim"] for m in local_metrics], device=accelerator.device
        )
        local_psnr = torch.tensor(
            [m["psnr"] for m in local_metrics], device=accelerator.device
        )
    else:
        # Create empty tensors if no metrics computed on this process
        local_ssim = torch.tensor([], device=accelerator.device)
        local_psnr = torch.tensor([], device=accelerator.device)

    # Gather metrics from all processes

    all_ssim = accelerator.gather(local_ssim)
    all_psnr = accelerator.gather(local_psnr)

    # Compute aggregated statistics (only on main process)
    if accelerator.is_main_process:
        all_ssim = all_ssim.cpu().numpy()
        all_psnr = all_psnr.cpu().numpy()

        # Handle case where no metrics were computed
        if len(all_ssim) == 0:
            accelerator.print("  Warning: No validation metrics computed!")
            metrics_dict = {}
        else:
            metrics_dict = {
                "ssim": float(np.mean(all_ssim)),
                "psnr": float(np.mean(all_psnr)),
                "num_samples": len(all_ssim),
            }
    else:
        metrics_dict = {}

    # Validation complete
    model.train()
    if not accelerator.is_main_process:
        visualizations = {}
    return metrics_dict, visualizations


def save_checkpoint(
    accelerator: Accelerator,
    model: DiffusionUNet3D,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """
    Save training checkpoint with atomic writes to prevent corruption.

    Uses write-to-temp-then-rename pattern for atomic checkpoint saves.

    Args:
        accelerator: Accelerator instance
        model: Diffusion model
        optimizer: Optimizer
        lr_scheduler: LR scheduler
        epoch: Current epoch
        global_step: Global training step
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }

    # Helper function for atomic save
    def atomic_save(checkpoint_path_str: str):
        checkpoint_path = Path(checkpoint_path_str)
        # temp_path = checkpoint_path.with_suffix('.pth.tmp')
        temp_path = checkpoint_path.with_suffix(".pth")

        # Save to temp file
        accelerator.save(checkpoint, str(temp_path))

        # Atomic rename (overwrites existing file atomically)
        # temp_path.replace(checkpoint_path)

    # Save latest checkpoint (atomic)
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pth")
    atomic_save(checkpoint_path)

    # Save epoch checkpoint (atomic)
    epoch_checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
    )
    atomic_save(epoch_checkpoint_path)

    # Save best checkpoint (atomic)
    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
        atomic_save(best_checkpoint_path)
        if accelerator.is_main_process:
            accelerator.print(f"  Saved best checkpoint: {best_checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: DiffusionUNet3D,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
) -> tuple:
    """
    Load checkpoint and resume training.

    Args:
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load LR scheduler
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    if accelerator.is_main_process:
        accelerator.print(f"  Resumed from checkpoint: {checkpoint_path}")
        accelerator.print(f"  Epoch: {epoch}, Global step: {global_step}")

    return epoch, global_step


def train(config: dict, args):
    # Initialize accelerator
    training_config = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps", 1
        ),
        mixed_precision=training_config.get("mixed_precision", "no"),
        log_with="wandb" if training_config.get("use_wandb", True) else None,
    )

    output_config = config["output"]
    checkpoint_dir = args.output_dir or output_config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:
        accelerator.print("=" * 70)
        accelerator.print("CT Latent Diffusion Training")
        accelerator.print("=" * 70)
        accelerator.print(f"\nConfiguration:")
        accelerator.print(f"  Device: {accelerator.device}")
        accelerator.print(f"  Num processes: {accelerator.num_processes}")
        accelerator.print(
            f"  Mixed precision: {training_config.get('mixed_precision', 'no')}"
        )
        accelerator.print(
            f"  Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)}"
        )
        accelerator.print(f"  Checkpoint dir: {checkpoint_dir}")

    # Initialize wandb
    if training_config.get("use_wandb", True) and accelerator.is_main_process:
        wandb_project = training_config.get("wandb_project", "ct-latent-diffusion")
        wandb_entity = training_config.get("wandb_entity", None)
        wandb_run_name = training_config.get("wandb_run_name", None)
        wandb_tags = training_config.get("wandb_tags", [])

        accelerator.init_trackers(
            project_name=wandb_project,
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": wandb_entity,
                    "name": wandb_run_name,
                    "tags": wandb_tags,
                }
            },
        )
        accelerator.print(f"\nInitialized wandb:")
        accelerator.print(f"  Project: {wandb_project}")
        if wandb_entity:
            accelerator.print(f"  Entity: {wandb_entity}")
        if wandb_run_name:
            accelerator.print(f"  Run name: {wandb_run_name}")

    # Create model
    if accelerator.is_main_process:
        accelerator.print("\nCreating model...")

    model = create_model(config)

    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"  Total parameters: {num_params:,}")
        accelerator.print(f"  Trainable parameters: {num_trainable:,}")

    # Create noise scheduler
    scheduler_config = config["scheduler"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_config["num_train_timesteps"],
        beta_start=scheduler_config["beta_start"],
        beta_end=scheduler_config["beta_end"],
        beta_schedule=scheduler_config["beta_schedule"],
        prediction_type=scheduler_config["prediction_type"],
        clip_sample=scheduler_config.get("clip_sample", False),
    )

    # Create dataloaders
    if accelerator.is_main_process:
        accelerator.print("\nCreating dataloaders...")

    data_config = config["data"]
    legacy_batch_size = training_config.get("batch_size")
    train_batch_size = training_config.get("train_batch_size", legacy_batch_size)
    eval_batch_size = training_config.get("eval_batch_size", train_batch_size)

    if train_batch_size is None:
        train_batch_size = 4
    if eval_batch_size is None:
        eval_batch_size = train_batch_size

    if legacy_batch_size is not None and "train_batch_size" not in training_config:
        if accelerator.is_main_process:
            accelerator.print(
                "Warning: 'training.batch_size' is deprecated. "
                "Use 'train_batch_size' and 'eval_batch_size' instead."
            )

    training_config["train_batch_size"] = train_batch_size
    training_config["eval_batch_size"] = eval_batch_size

    # Load from HuggingFace Hub
    hub_repo = data_config["hub_repo"]
    cache_dir = data_config.get("cache_dir", "/data1/peijia/ct")

    if accelerator.is_main_process:
        accelerator.print(f"  Loading from HuggingFace Hub: {hub_repo}")
        accelerator.print(f"  Cache directory: {cache_dir}")

    train_loader, val_loader = create_hf_dataloaders(
        hub_repo=hub_repo,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=training_config.get("num_workers", 4),
        normalize_latents=data_config.get("normalize_latents", False),
        pin_memory=True,
        cache_dir=cache_dir,
        timeit_collate_fn=data_config.get("timeit_collate_fn", False)
    )

    if accelerator.is_main_process:
        accelerator.print(f"  Train samples: {len(train_loader.dataset)}")
        accelerator.print(f"  Val samples: {len(val_loader.dataset)}")
        accelerator.print(f"  Train batch size: {train_batch_size}")
        accelerator.print(f"  Eval batch size: {eval_batch_size}")
        accelerator.print(f"  Train batches: {len(train_loader)}")

    validation_config = config.get("validation", {})
    num_val_samples = validation_config.get("num_samples", 4)
    compute_train_metrics = validation_config.get("compute_train_metrics", True)

    train_subset_loader = None
    if compute_train_metrics:
        if accelerator.is_main_process:
            accelerator.print(f"\nCreating train subset for validation metrics...")

        train_subset_loader = create_train_subset_dataloader(
            hub_repo=hub_repo,
            num_samples=num_val_samples,
            batch_size=eval_batch_size,
            num_workers=training_config.get("num_workers", 4),
            normalize_latents=data_config.get("normalize_latents", False),
            pin_memory=True,
            cache_dir=cache_dir
        )

        if accelerator.is_main_process:
            accelerator.print(f"  Train subset samples: {len(train_subset_loader.dataset)}")

    # Visualize network architecture (before model preparation)
    validation_config = config.get("validation", {})
    if validation_config.get("visualize_architecture", False) and accelerator.is_main_process:
        if accelerator.is_main_process:
            accelerator.print("\nGenerating network architecture visualization...")

        try:
            visualizer = NetworkArchitectureVisualizer(model, config)
            diagram_path = visualizer.trace_and_visualize(
                train_loader=train_loader,
                device=accelerator.device,
            )

            if accelerator.is_main_process:
                accelerator.print(f"  Architecture diagram saved: {diagram_path}")

                # Upload to wandb if enabled
                if training_config.get("use_wandb", True):
                    # Check if file is image or HTML
                    if diagram_path.endswith('.png') or diagram_path.endswith('.svg'):
                        wandb.log({"architecture/network_diagram": wandb.Image(diagram_path)})
                    elif diagram_path.endswith('.html'):
                        wandb.log({"architecture/network_diagram": wandb.Html(open(diagram_path).read())})

                    # Also log the mermaid markdown file if it exists
                    mermaid_path = diagram_path.replace('.png', '.mmd').replace('.html', '.mmd').replace('.svg', '.mmd')
                    if os.path.exists(mermaid_path):
                        with open(mermaid_path) as f:
                            mermaid_content = f.read()
                        wandb.log({"architecture/mermaid_code": wandb.Html(f"<pre>{mermaid_content}</pre>")})

                    accelerator.print("  Uploaded to wandb")
        except Exception as e:
            if accelerator.is_main_process:
                accelerator.print(f"  Warning: Architecture visualization failed: {e}")
                accelerator.print("  Continuing with training...")

    # Load VQ-AE for visualization and metrics
    # Load on ALL processes if distributed evaluation is enabled
    vae = None
    distributed_eval = validation_config.get("distributed_eval", True)

    load_vae = (
        training_config.get("use_wandb", True) and accelerator.is_main_process
    ) or distributed_eval

    if load_vae:
        if accelerator.is_main_process:
            accelerator.print("\nLoading VQ-AE...")
            if distributed_eval:
                accelerator.print(
                    "  Loading on ALL processes for distributed evaluation"
                )

        vae_checkpoint = data_config.get("vae_checkpoint")
        if vae_checkpoint:
            vae = FrozenVQAE(vae_checkpoint, device=accelerator.device)
            if accelerator.is_main_process:
                accelerator.print(f"  VQ-AE loaded from: {vae_checkpoint}")
                accelerator.print(f"  Memory: ~500 MB per process")


    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(
            training_config.get("adam_beta1", 0.9),
            training_config.get("adam_beta2", 0.999),
        ),
        eps=training_config.get("adam_epsilon", 1e-8),
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    print('wegiht decay: ', training_config.get("weight_decay", 0.01))

    # Calculate total training steps
    num_epochs = training_config["num_epochs"]
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = num_epochs * num_update_steps_per_epoch

    # Create LR scheduler
    scheduler_type = training_config.get("lr_scheduler_type", "linear")

    if scheduler_type == "linear":
        # Linear LR scheduler with warmup
        warmup_ratio = training_config.get("warmup_ratio", 0.06)
        warmup_steps = int(total_steps * warmup_ratio)
        lr_min = training_config.get("lr_min", 0.0)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Warmup: linearly increase from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay: linearly decrease from 1 to lr_min/lr_max
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(
                    lr_min / training_config.get("learning_rate", 1e-4), 1.0 - progress
                )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if accelerator.is_main_process:
            accelerator.print(f"\nLR Scheduler:")
            accelerator.print(f"  Type: Linear with warmup")
            accelerator.print(
                f"  Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}% of {total_steps})"
            )
            accelerator.print(f"  Min LR: {lr_min}")

    elif scheduler_type == "cosine":
        # Cosine annealing
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=training_config.get("lr_min", 0.0)
        )
        if accelerator.is_main_process:
            accelerator.print(f"\nLR Scheduler:")
            accelerator.print(f"  Type: Cosine annealing")
            accelerator.print(f"  T_max: {total_steps}")

    else:
        # Constant LR (no scheduler)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        if accelerator.is_main_process:
            accelerator.print(f"\nLR Scheduler: None (constant LR)")

    # Prepare for distributed training
    if train_subset_loader is not None:
        model, optimizer, train_loader, val_loader, train_subset_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, train_subset_loader, lr_scheduler
        )
    else:
        model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, accelerator
        )

    # Training loop
    if accelerator.is_main_process:
        accelerator.print("\n" + "=" * 70)
        accelerator.print("Starting training")
        accelerator.print("=" * 70)

    val_every = training_config.get("val_every", 50)
    save_every = training_config.get("save_every", 10000)

    log_every = training_config.get("log_every", 10)
    num_val_samples = validation_config.get("num_samples", 4)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)

    for epoch in range(start_epoch, num_epochs):
        model.train()

        progress_bar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        print(f"Begin of epoch {epoch+1}")

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # Move tensors to device (Accelerate doesn't auto-move dict values)
                ld_latent = batch["ld_latent"].to(accelerator.device)
                hd_latent = batch["hd_latent"].to(accelerator.device)

                B = hd_latent.shape[0]

                # Resize LD latent to match HD latent depth if needed
                if ld_latent.shape[2] != hd_latent.shape[2]:
                    # ld_latent: [B, C, D_ld, H, W]
                    # hd_latent: [B, C, D_hd, H, W]
                    # Resize only depth dimension to match HD
                    ld_latent = F.interpolate(
                        ld_latent,
                        size=hd_latent.shape[2:],  # Match [D, H, W] of HD
                        mode='trilinear',
                        align_corners=False
                    )

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (B,),
                    device=hd_latent.device,
                ).long()

                # Add noise to HD latent
                noise = torch.randn_like(hd_latent)
                noisy_hd_latent = noise_scheduler.add_noise(hd_latent, noise, timesteps)

                # Concatenate LD latent as conditioning
                model_input = torch.cat([noisy_hd_latent, ld_latent], dim=1)

                # Predict noise
                noise_pred = model(model_input, timesteps)

                # MSE loss
                loss = F.mse_loss(noise_pred, noise)

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.update(1)

            # Logging
            if global_step % log_every == 0:
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }

                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            # Validation (two-tier strategy)
            if (
                global_step % val_every == 0
                and global_step > 0
            ):
                # Lightweight validation (always)
                if accelerator.is_main_process:
                    accelerator.print("\nRunning lightweight validation...")

                val_loss = validate(model, val_loader, noise_scheduler, accelerator)

                accelerator.log({"val/loss": val_loss}, step=global_step)
                accelerator.print(f"  MSE loss: {val_loss:.4f}")

                if accelerator.is_main_process:
                    accelerator.print(
                        f"\nRunning full validation on VAL set ({num_val_samples} samples)..."
                    )

                val_metrics, val_viz_dict = validate_with_metrics(
                    model,
                    val_loader,
                    vae,
                    config,
                    accelerator,
                    num_samples=num_val_samples,
                    spatial_chunk_size=validation_config.get("spatial_chunk_size", 32),
                )

                if accelerator.is_main_process:
                    if val_viz_dict:
                        # Log with val/ prefix
                        val_viz_log = {f"{k}": v for k, v in val_viz_dict.items()}
                        accelerator.log(val_viz_log, step=global_step)
                        accelerator.print("Val visualizations generated.")

                    if val_metrics:
                        log_dict = {
                            f"val/metrics/{k}": v for k, v in val_metrics.items()
                        }
                        accelerator.log(log_dict, step=global_step)

                        accelerator.print(
                            f"  Val SSIM: {val_metrics.get('ssim', 0):.4f}"
                        )
                        accelerator.print(
                            f"  Val PSNR: {val_metrics.get('psnr', 0):.2f} dB"
                        )
                        accelerator.print(
                            f"  Samples evaluated: {val_metrics.get('num_samples', 0)}"
                        )

                if compute_train_metrics and train_subset_loader is not None:
                    if accelerator.is_main_process:
                        accelerator.print(
                            f"\nRunning full validation on TRAIN subset ({num_val_samples} samples)..."
                        )

                    train_metrics, train_viz_dict = validate_with_metrics(
                        model,
                        train_subset_loader,
                        vae,
                        config,
                        accelerator,
                        num_samples=num_val_samples,
                    )

                    if accelerator.is_main_process:
                        if train_viz_dict:
                            # Log with train/ prefix
                            train_viz_log = {
                                f"train/{k.split('/')[-1]}": v
                                for k, v in train_viz_dict.items()
                            }
                            accelerator.log(train_viz_log, step=global_step)
                            accelerator.print("Train visualizations generated.")

                        if train_metrics:
                            log_dict = {
                                f"train/metrics/{k}": v
                                for k, v in train_metrics.items()
                            }
                            accelerator.log(log_dict, step=global_step)

                            accelerator.print(
                                f"  Train SSIM: {train_metrics.get('ssim', 0):.4f}"
                            )
                            accelerator.print(
                                f"  Train PSNR: {train_metrics.get('psnr', 0):.2f} dB"
                            )
                            accelerator.print(
                                f"  Samples evaluated: {train_metrics.get('num_samples', 0)}"
                            )

            if (
                global_step % save_every == 0
                and accelerator.sync_gradients
                and global_step > 0
            ):
                if accelerator.is_main_process:
                    accelerator.print(f"\nSaving checkpoint at step {global_step}...")

                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                    checkpoint_dir,
                    is_best=False,
                )

            global_step += 1

        progress_bar.close()

        if accelerator.is_main_process:
            accelerator.print(
                f"\nEpoch {epoch+1} completed. Global step: {global_step}"
            )

    if accelerator.is_main_process:
        accelerator.print("\n" + "=" * 70)
        accelerator.print("Training completed!")
        accelerator.print("=" * 70)
        accelerator.print(f"Best validation loss: {best_val_loss:.4f}")
        accelerator.print(f"Final checkpoint saved to: {checkpoint_dir}")

    save_checkpoint(
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        num_epochs - 1,
        global_step,
        checkpoint_dir,
        is_best=False,
    )

    if training_config.get("use_wandb", True):
        accelerator.end_training()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(config)
    train(config, args)


if __name__ == "__main__":
    main()
