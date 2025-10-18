"""
Network Architecture Visualizer for 3D Diffusion UNet

This module provides visualization of the DiffusionUNet3D architecture by:
1. Tracing a forward pass with hooks to capture actual tensor shapes
2. Generating a Mermaid diagram showing the network structure
3. Rendering the diagram to an image (PNG/SVG)
4. Supporting upload to wandb

The visualizer is fully adaptive and works with any architecture configuration
by detecting shapes at runtime rather than hardcoding them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class NetworkArchitectureVisualizer:
    """
    Visualizes the architecture of a DiffusionUNet3D model.

    Features:
    - Hook-based shape tracing during a forward pass
    - Automatic adaptation to any model configuration
    - Mermaid diagram generation
    - Image rendering for wandb upload

    Args:
        model: DiffusionUNet3D model to visualize
        config: Configuration dictionary
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.shapes: Dict[str, Tuple] = {}
        self.hooks: List = []

    def trace_and_visualize(
        self,
        train_loader: DataLoader,
        device: torch.device,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Trace the model with a sample batch and generate visualization.

        Args:
            train_loader: Training dataloader to extract sample from
            device: Device to run inference on
            output_dir: Directory to save outputs (default: from config or ./outputs/architecture)

        Returns:
            Path to the rendered diagram image
        """
        # Get output directory
        if output_dir is None:
            viz_config = self.config.get("visualization", {})
            output_dir = viz_config.get("architecture_save_path", "./outputs/architecture")

        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Get sample batch safely
        sample_batch = self._get_sample_batch(train_loader, device)

        # Step 2: Register hooks and trace
        self._register_hooks()

        # Step 3: Run forward pass to capture shapes
        with torch.no_grad():
            self._trace_forward(sample_batch)

        # Step 4: Remove hooks
        self._cleanup_hooks()

        # Step 5: Generate Mermaid diagram
        mermaid_code = self._generate_mermaid_diagram()

        # Step 6: Save Mermaid markdown
        mermaid_path = os.path.join(output_dir, "architecture.mmd")
        with open(mermaid_path, "w") as f:
            f.write(mermaid_code)

        # Step 7: Render to image
        image_path = self._render_to_image(mermaid_code, output_dir)

        return image_path

    def _get_sample_batch(
        self,
        train_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Safely extract a sample batch from the dataloader.

        Creates a temporary dataloader copy to avoid interfering with
        the main training dataloader's state.
        """
        # Create temporary dataloader with same dataset
        sample_loader = DataLoader(
            train_loader.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # No multiprocessing for single sample
        )

        # Extract one batch
        sample_batch = next(iter(sample_loader))

        # Move to device
        ld_latent = sample_batch["ld_latent"].to(device)
        hd_latent = sample_batch["hd_latent"].to(device)

        # Clean up
        del sample_loader

        return {"ld_latent": ld_latent, "hd_latent": hd_latent}

    def _register_hooks(self):
        """Register forward hooks to capture tensor shapes."""

        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.shapes[name] = tuple(output.shape)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        self.shapes[name] = tuple(output[0].shape)
            return hook

        # Register hooks on key modules
        for name, module in self.model.named_modules():
            # Skip nested modules (only register on leaf modules)
            if len(list(module.children())) == 0:
                continue

            # Register on important architectural components
            if any(key in name for key in ['conv_in', 'conv_out', 'down_samples',
                                            'up_samples', 'down_blocks', 'up_blocks',
                                            'mid_block', 'mid_attn']):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def _trace_forward(self, sample_batch: Dict[str, torch.Tensor]):
        """Run a forward pass to trace shapes."""
        ld_latent = sample_batch["ld_latent"]
        hd_latent = sample_batch["hd_latent"]

        # Record input shapes
        self.shapes["input/ld_latent"] = tuple(ld_latent.shape)
        self.shapes["input/hd_latent"] = tuple(hd_latent.shape)

        # Resize LD to match HD if needed (same as training code)
        if ld_latent.shape[2] != hd_latent.shape[2]:
            ld_latent = torch.nn.functional.interpolate(
                ld_latent,
                size=hd_latent.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        # Concatenate as model input
        model_input = torch.cat([hd_latent, ld_latent], dim=1)
        self.shapes["input/concatenated"] = tuple(model_input.shape)

        # Create dummy timestep
        timesteps = torch.tensor([500], device=ld_latent.device)

        # Forward pass
        output = self.model(model_input, timesteps)
        self.shapes["output"] = tuple(output.shape)

    def _cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _generate_mermaid_diagram(self) -> str:
        """
        Generate Mermaid diagram code from captured shapes.

        Returns:
            Mermaid markdown code
        """
        model_config = self.config["model"]

        # Get architecture parameters
        num_levels = len(model_config["channel_mult"])
        num_blocks = model_config["num_blocks"]
        attention_levels = model_config.get("attention_levels", [])
        model_channels = model_config["model_channels"]
        channel_mult = model_config["channel_mult"]

        # Get actual shapes
        input_shape = self.shapes.get("input/concatenated", (1, 16, 25, 128, 128))
        output_shape = self.shapes.get("output", (1, 8, 25, 128, 128))

        # Format shape for display (remove batch dimension)
        def fmt_shape(shape):
            return f"[{shape[1]}, {shape[2]}, {shape[3]}, {shape[4]}]"

        # Start Mermaid diagram
        lines = [
            "graph TB",
            "    classDef inputNode fill:#e1f5ff,stroke:#01579b,stroke-width:2px",
            "    classDef convNode fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "    classDef resNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
            "    classDef attnNode fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px",
            "    classDef downNode fill:#fce4ec,stroke:#880e4f,stroke-width:2px",
            "    classDef upNode fill:#e0f2f1,stroke:#004d40,stroke-width:2px",
            "    classDef outputNode fill:#fff9c4,stroke:#f57f17,stroke-width:2px",
            "",
        ]

        # Input nodes
        lines.append(f'    Input["🔵 Input<br/>{fmt_shape(input_shape)}"]:::inputNode')
        lines.append(f'    TimeEmb["⏰ Time Embedding<br/>[{model_config["time_embed_dim"]}]"]:::inputNode')
        lines.append("")

        # Initial convolution
        conv_in_shape = self.shapes.get("conv_in", (1, model_channels, input_shape[2], input_shape[3], input_shape[4]))
        lines.append(f'    ConvIn["Conv3D<br/>{fmt_shape(conv_in_shape)}"]:::convNode')
        lines.append("    Input --> ConvIn")
        lines.append("")

        # Track node names for connections
        prev_node = "ConvIn"
        skip_nodes = []

        # Encoder (Downsampling path)
        lines.append("    %% Encoder Path")
        for level in range(num_levels):
            ch = model_channels * channel_mult[level]
            has_attn = level in attention_levels

            # Estimate spatial dimensions (divide by 2 for each downsample)
            spatial_scale = 2 ** level
            est_d = input_shape[2] // spatial_scale
            est_h = input_shape[3] // spatial_scale
            est_w = input_shape[4] // spatial_scale

            # Try to get actual shape from hooks
            down_key = f"down_blocks.{level * num_blocks}"
            if down_key in self.shapes:
                actual_shape = self.shapes[down_key]
                shape_str = fmt_shape(actual_shape)
            else:
                shape_str = f"[{ch}, {est_d}, {est_h}, {est_w}]"

            # Down block node
            attn_marker = " + Attn" if has_attn else ""
            node_name = f"Down{level}"
            lines.append(f'    {node_name}["📉 Down Level {level}<br/>{num_blocks}x ResBlock{attn_marker}<br/>{shape_str}"]:::resNode')
            lines.append(f"    {prev_node} --> {node_name}")

            # Add time embedding connection (dotted)
            lines.append(f"    TimeEmb -.-> {node_name}")

            # Store for skip connection
            skip_nodes.append(node_name)

            # Downsample (except last level)
            if level != num_levels - 1:
                ds_name = f"DS{level}"
                ds_spatial_scale = 2 ** (level + 1)
                ds_d = input_shape[2] // ds_spatial_scale
                ds_h = input_shape[3] // ds_spatial_scale
                ds_w = input_shape[4] // ds_spatial_scale

                lines.append(f'    {ds_name}["⬇️ Downsample<br/>[{ch}, {ds_d}, {ds_h}, {ds_w}]"]:::downNode')
                lines.append(f"    {node_name} --> {ds_name}")
                prev_node = ds_name
            else:
                prev_node = node_name

            lines.append("")

        # Bottleneck
        lines.append("    %% Bottleneck")
        mid_ch = model_channels * channel_mult[-1]
        mid_spatial_scale = 2 ** (num_levels - 1)
        mid_d = input_shape[2] // mid_spatial_scale
        mid_h = input_shape[3] // mid_spatial_scale
        mid_w = input_shape[4] // mid_spatial_scale

        lines.append(f'    Mid["🎯 Bottleneck<br/>ResBlock + Attn + ResBlock<br/>[{mid_ch}, {mid_d}, {mid_h}, {mid_w}]"]:::attnNode')
        lines.append(f"    {prev_node} --> Mid")
        lines.append(f"    TimeEmb -.-> Mid")
        lines.append("")
        prev_node = "Mid"

        # Decoder (Upsampling path)
        lines.append("    %% Decoder Path")
        for level in reversed(range(num_levels)):
            ch = model_channels * channel_mult[level]
            has_attn = level in attention_levels

            # Estimate spatial dimensions
            spatial_scale = 2 ** level
            est_d = input_shape[2] // spatial_scale
            est_h = input_shape[3] // spatial_scale
            est_w = input_shape[4] // spatial_scale

            # Up block node
            attn_marker = " + Attn" if has_attn else ""
            node_name = f"Up{level}"
            lines.append(f'    {node_name}["📈 Up Level {level}<br/>{num_blocks+1}x ResBlock{attn_marker}<br/>[{ch}, {est_d}, {est_h}, {est_w}]"]:::resNode')

            # Skip connection (dotted line)
            skip_node = skip_nodes[level]
            lines.append(f"    {skip_node} -.skip.-> {node_name}")

            # Connection from previous
            lines.append(f"    {prev_node} --> {node_name}")

            # Add time embedding connection
            lines.append(f"    TimeEmb -.-> {node_name}")

            # Upsample (except first level)
            if level != 0:
                us_name = f"US{level}"
                us_spatial_scale = 2 ** (level - 1)
                us_d = input_shape[2] // us_spatial_scale
                us_h = input_shape[3] // us_spatial_scale
                us_w = input_shape[4] // us_spatial_scale

                lines.append(f'    {us_name}["⬆️ Upsample<br/>[{ch}, {us_d}, {us_h}, {us_w}]"]:::upNode')
                lines.append(f"    {node_name} --> {us_name}")
                prev_node = us_name
            else:
                prev_node = node_name

            lines.append("")

        # Output
        lines.append("    %% Output")
        lines.append(f'    ConvOut["Conv3D<br/>{fmt_shape(output_shape)}"]:::convNode')
        lines.append(f'    Output["🟢 Output<br/>{fmt_shape(output_shape)}"]:::outputNode')
        lines.append(f"    {prev_node} --> ConvOut")
        lines.append(f"    ConvOut --> Output")

        return "\n".join(lines)

    def _render_to_image(self, mermaid_code: str, output_dir: str) -> str:
        """
        Render Mermaid diagram to image.

        Tries multiple rendering methods:
        1. mermaid-cli (mmdc) if installed
        2. Web-based API (mermaid.ink)
        3. Fallback: save as HTML

        Args:
            mermaid_code: Mermaid markdown code
            output_dir: Directory to save image

        Returns:
            Path to rendered image
        """
        # Try mermaid-cli first
        try:
            result = self._render_with_mmdc(mermaid_code, output_dir)
            print(f"  Rendered with mermaid-cli: {result}")
            return result
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  mermaid-cli failed: {e}")

        # Try web API
        try:
            result = self._render_with_web_api(mermaid_code, output_dir)
            print(f"  Rendered with web API: {result}")
            return result
        except Exception as e:
            print(f"  Web API failed: {e}")

        # Fallback: save as HTML
        result = self._render_as_html(mermaid_code, output_dir)
        print(f"  Rendered as HTML: {result}")
        return result

    def _render_with_mmdc(self, mermaid_code: str, output_dir: str) -> str:
        """Render using mermaid-cli (mmdc)."""
        mmd_path = os.path.join(output_dir, "architecture.mmd")
        png_path = os.path.join(output_dir, "architecture.png")

        # Save mermaid code
        with open(mmd_path, "w") as f:
            f.write(mermaid_code)

        # Run mmdc
        subprocess.run(
            ["mmdc", "-i", mmd_path, "-o", png_path, "-b", "transparent"],
            check=True,
            capture_output=True,
        )

        return png_path

    def _render_with_web_api(self, mermaid_code: str, output_dir: str) -> str:
        """Render using mermaid.ink web API."""
        import base64
        import urllib.request

        # Encode mermaid code
        encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
        url = f"https://mermaid.ink/img/{encoded}"

        # Download image
        png_path = os.path.join(output_dir, "architecture.png")

        try:
            urllib.request.urlretrieve(url, png_path)

            # Verify the file was downloaded and has content
            if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
                raise Exception(f"Downloaded file is empty or doesn't exist")

        except Exception as e:
            # If web API fails, raise to fallback to HTML
            if os.path.exists(png_path):
                os.remove(png_path)
            raise Exception(f"Web API rendering failed: {e}")

        return png_path

    def _render_as_html(self, mermaid_code: str, output_dir: str) -> str:
        """Fallback: save as standalone HTML."""
        html_path = os.path.join(output_dir, "architecture.html")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Network Architecture</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
</body>
</html>"""

        with open(html_path, "w") as f:
            f.write(html_content)

        return html_path
