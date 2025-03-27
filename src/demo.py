import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from pathlib import Path

from src.model.encoder.encoder_noposplat import EncoderNoPoSplat, EncoderNoPoSplatCfg, OpacityMappingCfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from model.ply_export import export_ply  # Assuming this function exists

# Dummy config (replace with custom config loading if needed)
vis_cfg = EncoderVisualizerEpipolarCfg(num_samples=8, min_resolution=256, export_ply=False)
gs_cfg = GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4)
om_cfg = OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1)

cfg = EncoderNoPoSplatCfg(
    name="noposplat",
    d_feature=128,
    num_monocular_samples=32,
    backbone='croco',
    visualizer=vis_cfg,
    gaussian_adapter=gs_cfg,
    apply_bounds_shim=False,
    opacity_mapping=om_cfg,
    gaussians_per_pixel=1,
    num_surfaces=1,
    gs_params_head_type="dpt_gs",
)

encoder = EncoderNoPoSplat(cfg)
encoder.eval()

root_path = Path('/workspace/NoPoSplat')

# Load checkpoint
try:
    ckpt_path = root_path / "mixRe10kDl3dv.ckpt"  # Replace with your path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    global_step = ckpt["global_step"]
    ckpt_weights = ckpt["state_dict"]
    ckpt_weights = {k[8:] if k.startswith("encoder.") else k: v for k, v in ckpt_weights.items()}
    encoder.load_state_dict(ckpt_weights, strict=True)
except:
    raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

# Load and preprocess 2 images
def load_image(path):
    image = read_image(str(path)).float() / 255.0  # [C,H,W]
    image = resize(image, [256, 256])  # Resize to match model input(256 or 512), if needed
    return image.to("cuda")  # Move to designated CUDA device

img1 = load_image(root_path / 'demo_images' / 'sample1.png')
img2 = load_image(root_path / 'demo_images' / 'sample2.png')
images = torch.stack([img1, img2], dim=0).unsqueeze(0).to("cuda")  # [B=1, V=2, C, H, W]

# Replace intrinsics/extrinsics if evaluating on real data
context = {
    "image": images,
    "intrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
    "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1),
}

with torch.no_grad():
    gaussians = encoder(context, global_step)

# Define output path
ply_path = Path(root_path / 'outputs' / 'image_collections/demo_output.ply')
ply_path.parent.mkdir(parents=True, exist_ok=True)

# Export to .ply
export_ply(gaussians.means, gaussians.scales, gaussians.rotations, gaussians.harmonics, gaussians.opacities, ply_path)
print(f"PLY file written to {ply_path}")