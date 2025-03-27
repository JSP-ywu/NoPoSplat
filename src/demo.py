import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from pathlib import Path
from einops import rearrange  # Add this at the top if not already present

from src.model.encoder.encoder_noposplat import EncoderNoPoSplat, EncoderNoPoSplatCfg, OpacityMappingCfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from src.model.ply_export import export_ply
from src.model.encoder.visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

# vis on? off?
vis = True


# Dummy config (replace with custom config loading if needed)
print('Dummy config generation....')
vis_cfg = EncoderVisualizerEpipolarCfg(num_samples=8, min_resolution=256, export_ply=False)
gs_cfg = GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4)
om_cfg = OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1)
bb_cfg = BackboneCrocoCfg(name="croco", model='ViTLarge_BaseDecoder', patch_embed_cls='PatchEmbedDust3R',
                          asymmetry_decoder=True, intrinsics_embed_degree=4, intrinsics_embed_loc='encoder', intrinsics_embed_type='token')

cfg = EncoderNoPoSplatCfg(
    name="noposplat",
    d_feature=128,
    num_monocular_samples=32,
    backbone=bb_cfg,
    visualizer=vis_cfg,
    gaussian_adapter=gs_cfg,
    apply_bounds_shim=False,
    opacity_mapping=om_cfg,
    gaussians_per_pixel=1,
    num_surfaces=1,
    gs_params_head_type="dpt_gs",
    demo = True
)
print('Done!')

if vis:
    print('Setting visualizer....')
    visualizer = EncoderVisualizerEpipolar(vis_cfg)
    print('Done!')

print('Loading encoder and pretrained weights....')
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
    encoder.load_state_dict(ckpt_weights, strict=False)
except:
    raise ValueError(f"Invalid checkpoint format: {ckpt_path}")
encoder.to('cuda')
print('Done!')

# Load and preprocess 2 images
def load_image(path):
    image = read_image(str(path)).float() / 255.0  # [C,H,W]
    image = resize(image, [256, 256])  # Resize to match model input(256 or 512), if needed
    return image.to("cuda")  # Move to designated CUDA device

print('Loading images....')
img1 = load_image(root_path / 'demo_images' / '30331971_8549387056.jpg')
img2 = load_image(root_path / 'demo_images' / '30496019_3709943269.jpg')
images = torch.stack([img1, img2], dim=0).unsqueeze(0).to("cuda")  # [B=1, V=2, C, H, W]
print('Done!')
# Replace intrinsics/extrinsics with shapes matching convert_poses logic
intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1).to("cuda")  # [B, V, 3, 3]
extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1).to("cuda")  # [B, V, 4, 4]

context = {
    "image": images,
    "intrinsics": intrinsics,
    "extrinsics": extrinsics,
}

# print("intrinsics device:", context["intrinsics"].device)
# print("image device:", context["image"].device)

print('Inference....')
with torch.no_grad():
    gaussians, gaussians_scales, gaussians_rotations = encoder(context, global_step)

# Flatten view/ray/surface/sample dimensions
gaussians_means = rearrange(gaussians.means, "b (gaussians) xyz -> (b gaussians) xyz")
gaussians_scales = rearrange(gaussians_scales, "b (gaussians) xyz -> (b gaussians) xyz")
gaussians_rotations = rearrange(gaussians_rotations, "b (gaussians) quat -> (b gaussians) quat")
gaussians_harmonics = rearrange(gaussians.harmonics, "b (gaussians) xyz d_sh -> (b gaussians) xyz d_sh")
gaussians_opacities = rearrange(gaussians.opacities, "b (gaussians) -> (b gaussians)")

print('Done!')

# Define output path
print('Exporting to PLY....')
ply_path = Path(root_path / 'outputs' / 'image_collections'/ 'demo_output_1.ply')
ply_path.parent.mkdir(parents=True, exist_ok=True)

# Export to .ply
export_ply(gaussians_means, gaussians_scales, gaussians_rotations, gaussians_harmonics, gaussians_opacities, ply_path)
print(f"PLY file written to {ply_path}")

# Visualize
import torchvision.transforms.functional as TF
from PIL import Image

if vis:
    print('Start gaussian visualization....')
    vis_image = visualizer.visualize(
        context['image'],
        gaussians.opacities,
        gaussians.covariances,
        gaussians.harmonics[..., 0],
    )
    vis_image = vis_image.detach().cpu().clamp(0, 1)

    img = TF.to_pil_image(vis_image)
    img.save(root_path / 'outputs' / 'image_collections' / 'demo_visualization_1.png')
    print('Done!')