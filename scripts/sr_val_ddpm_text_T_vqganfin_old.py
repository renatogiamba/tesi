"""make variations of input image"""

from pytorch_wavelets import DWTForward
import torch.nn.functional as F
import argparse, os, sys, glob, csv
import PIL
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from ldm.modules.diffusionmodules.util import linear,timestep_embedding

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	#change time embed
	from ldm.modules.diffusionmodules.util import linear
	from torch import nn
	try:
		if config.model.params.ship_embedder_config: #crasha sempre se non ci sta ma va bene cosi tanto ci sta l except
			model_channels = config.model.params.structcond_stage_config.params.model_channels
			time_embed_dim =  model_channels * 2 #changed from 4 to 2 for class embed
			model.structcond_stage_model.time_embed = nn.Sequential(
				linear(model_channels, time_embed_dim),
				nn.SiLU(),
				linear(time_embed_dim, time_embed_dim),
			)
	except:
		print('No time embed')
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def extract_info_from_csv(csv_file, mini_dict):
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        # mini_dict = {}
        next(csv_reader)

        for row in csv_reader:
            mini_dict[row[0]] = {}
            # mini_dict = row[0]
            mini_dict[row[0]]['gsd'] = row[1]
            mini_dict[row[0]]['cloud_cover'] = row[2]
            mini_dict[row[0]]['year'] = row[3]
            mini_dict[row[0]]['month'] = row[4]
            mini_dict[row[0]]['day'] = row[5]
            mini_dict[row[0]]['text_prompt'] = row[6]
            line_count += 1

        # print(f'Processed {line_count} lines.')
        return mini_dict


#function to list all csv files in path and subpath
def list_all_csv_files(path):
    ls = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                ls.append(os.path.join(root, file))
    return ls

def get_all_csv_info_as_dict(path):
    csvs = list_all_csv_files(path)
    diz = {}
    for i in range(len(csvs)):
        diz = extract_info_from_csv(csvs[i], diz)
    return diz

def get_metadata(meta_info, index):
    index = str(index)
    gsd = meta_info[index]['gsd']
    cloud_cover = meta_info[index]['cloud_cover']
    year = meta_info[index]['year']
    month = meta_info[index]['month']
    day = meta_info[index]['day']
    text_prompt = meta_info[index]['text_prompt']
    return gsd, cloud_cover, year, month, day, text_prompt


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload",
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=1000,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=1,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/model.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="models/ldm/stable-diffusion-v1/epoch=000011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)

	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w

	seed_everything(opt.seed)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.to(device)

	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples

	img_list_ori = os.listdir(opt.init_img)
	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	for item in img_list_ori:
		if os.path.exists(os.path.join(outpath, item)):
			img_list.remove(item)
			continue
		cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
		cur_image = transform(cur_image)
		cur_image = cur_image.clamp(-1, 1)
		init_image_list.append(cur_image)
	init_image_list = torch.cat(init_image_list, dim=0)
	niters = math.ceil(init_image_list.size(0) / batch_size)
	init_image_list = init_image_list.chunk(niters)

	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)
	meta_info = get_all_csv_info_as_dict("/content/tesi/content/test_csv")

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				all_samples = list()
				for n in trange(niters, desc="Sampling"):
					gsd, cloud_cover, year, month, day, text_prompt = get_metadata(meta_info,n)
					init_image = init_image_list[n]
					init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
					init_latent = model.get_first_stage_encoding(init_latent_generator)
					#text_init = ['']*init_image.size(0)
					text_init = text_prompt
					semantic_c = model.cond_stage_model(text_init)
					try:
						dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
						x_srll, x_srh = dwt(init_image)  
						x_srlh, x_srhl, x_srhh = torch.unbind(x_srh[0], dim=2)
						x_waves = torch.cat([x_srll, x_srlh, x_srhl, x_srhh], dim=1)
						x_waves = F.interpolate(x_waves, size=(224,224), mode='bilinear', align_corners=False)
						with torch.no_grad():
							x_waves = torch.stack([model.Vit_model.encode_image(sub_x_waves) for sub_x_waves in x_waves.split(3, dim=1)], dim=0).cuda()
							x_waves = x_waves.reshape(-1,4,512)
							batch_size = x_waves.shape[0]
							wavelet_embeddings = []
							for i in range(batch_size):
								t = x_waves[i]
								wav_embed = model.wav_emb(t.type(torch.float32))
								wavelet_embeddings.append(wav_embed)
						wavelet_embeddings = torch.cat(wavelet_embeddings, dim=0)
	
						model_channels = 256
						#time_embed_dim = model_channels * 4
						#time_embed = nn.Sequential(
            			#  		linear(model_channels, time_embed_dim),
            			#  		nn.SiLU(),
            			#  		linear(time_embed_dim, time_embed_dim),
            			#		)
						#time_embed.to(device)

						gsd_t = torch.tensor([float(gsd)]).to(device).long()
						gsd_emb = model.meta_emb(timestep_embedding(gsd_t, model_channels))
						cloud_t = torch.tensor([float(cloud_cover)]).to(device).long()
						cloud_emb = model.meta_emb(timestep_embedding(cloud_t, model_channels))
						year_t = torch.tensor([float(year)]).to(device).long()
						year_emb = model.meta_emb(timestep_embedding(year_t, model_channels))
						month_t = torch.tensor([float(month)]).to(device).long()
						month_emb = model.meta_emb(timestep_embedding(month_t, model_channels))
						day_t = torch.tensor([float(day)]).to(device).long()
						day_emb = model.meta_emb(timestep_embedding(day_t, model_channels))
						metadata_embeddings = gsd_emb + cloud_emb + year_emb + month_emb + day_emb
						#classes_embed = model.ship_label_emb(model.ship_classifier(init_image))

						wave_meta_embeddings = torch.cat([wavelet_embeddings,metadata_embeddings],dim=1)
					except:
						classes_embed = None
						print("No class embed")
					noise = torch.randn_like(init_latent)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
					t = t.to(device).long()
					x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
					x_T = None

					samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), 
							   timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True,
							   classes_embed=wave_meta_embeddings)
					x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, init_image)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, init_image)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

					for i in range(init_image.size(0)):
						img_name = img_list.pop(0)
						basename = os.path.splitext(os.path.basename(img_name))[0]
						x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(outpath, basename+'.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()