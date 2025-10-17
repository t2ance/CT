#!/bin/bash
# Upload pre-computed latents and VQ-AE checkpoint to HuggingFace Hub
# This is a ONE-TIME operation done from your local machine

set -e

echo "========================================="
echo "Upload CT Latents to HuggingFace Hub"
echo "========================================="

# Configuration
HF_USERNAME="t2ance"  # Change to your HuggingFace username
REPO_LATENTS="ct-latents"
REPO_VQAE="vqae-ct"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found!"
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

# Check if logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "Error: Not logged in to HuggingFace!"
    echo "Login with: huggingface-cli login"
    exit 1
fi

echo ""
echo "Logged in as: $(huggingface-cli whoami | grep username | awk '{print $2}')"
echo ""

# Upload latents (choose resolution)
read -p "Which latent cache to upload? (128/256/512): " RESOLUTION

LATENT_DIR="latents_cache_${RESOLUTION}"

if [ ! -d "$LATENT_DIR" ]; then
    echo "Error: Directory $LATENT_DIR not found!"
    exit 1
fi

echo ""
echo "Uploading $LATENT_DIR..."
echo "  Repo: ${HF_USERNAME}/${REPO_LATENTS}-${RESOLUTION}"
echo ""

# Calculate size
TOTAL_SIZE=$(du -sh "$LATENT_DIR" | awk '{print $1}')
echo "Total size: $TOTAL_SIZE"

read -p "Proceed with upload? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Upload cancelled."
    exit 0
fi

# Create dataset repo and upload
echo ""
echo "Creating/updating dataset repo..."
huggingface-cli repo create "${REPO_LATENTS}-${RESOLUTION}" --type dataset --exist-ok || true

echo ""
echo "Uploading latents..."
huggingface-cli upload \
    "${HF_USERNAME}/${REPO_LATENTS}-${RESOLUTION}" \
    "$LATENT_DIR" \
    --repo-type dataset \
    --commit-message "Upload pre-computed VQ-AE latents (${RESOLUTION}x${RESOLUTION}x${RESOLUTION})"

echo ""
echo "✓ Latents uploaded successfully!"
echo ""
echo "Dataset URL: https://huggingface.co/datasets/${HF_USERNAME}/${REPO_LATENTS}-${RESOLUTION}"

# Upload VQ-AE checkpoint
echo ""
echo "========================================="
echo "Upload VQ-AE Checkpoint"
echo "========================================="

VQAE_CHECKPOINT="$HOME/projects/BioAgent/3D-MedDiffusion/checkpoints/3DMedDiffusion_checkpoints/PatchVolume_8x_s2.ckpt"

if [ ! -f "$VQAE_CHECKPOINT" ]; then
    echo "Warning: VQ-AE checkpoint not found at: $VQAE_CHECKPOINT"
    echo "Skipping checkpoint upload."
    echo ""
    echo "To upload manually:"
    echo "  huggingface-cli upload ${HF_USERNAME}/${REPO_VQAE} /path/to/PatchVolume_8x_s2.ckpt --repo-type model"
    exit 0
fi

read -p "Upload VQ-AE checkpoint? (y/n): " CONFIRM_VQAE
if [ "$CONFIRM_VQAE" != "y" ]; then
    echo "Skipping VQ-AE checkpoint upload."
    exit 0
fi

echo ""
echo "Creating/updating model repo..."
huggingface-cli repo create "${REPO_VQAE}" --type model --exist-ok || true

echo ""
echo "Uploading VQ-AE checkpoint..."
huggingface-cli upload \
    "${HF_USERNAME}/${REPO_VQAE}" \
    "$VQAE_CHECKPOINT" \
    --repo-type model \
    --commit-message "Upload VQ-AE checkpoint (PatchVolume 8x compression)"

echo ""
echo "✓ VQ-AE checkpoint uploaded successfully!"
echo ""
echo "Model URL: https://huggingface.co/models/${HF_USERNAME}/${REPO_VQAE}"

# Summary
echo ""
echo "========================================="
echo "Upload Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Update ct-128-improved.yaml:"
echo "   - Replace YOUR_USERNAME with: ${HF_USERNAME}"
echo "   - Update repo names if different"
echo ""
echo "2. Deploy to K8s:"
echo "   kubectl apply -f ct-128-improved.yaml"
echo ""
echo "3. Monitor training:"
echo "   kubectl logs -f ct-128-training"
echo ""
