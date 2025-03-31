import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import PIL.Image

import node_helpers
from .resampler import Resampler
from .utils import tensor_to_image, resize_and_pad_image

from insightface.app import FaceAnalysis
from insightface.utils import face_align
from facexlib.recognition import init_recognition_model

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F

MODELS_DIR = os.path.join(folder_paths.models_dir, "InfiniteYou")
if "InfiniteYou" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["InfiniteYou"]
if os.path.exists(folder_paths.cache_dir):
    current_paths.append(os.path.join(folder_paths.cache_dir, "InfiniteYou"))
folder_paths.folder_names_and_paths["InfiniteYou"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
if os.path.exists(folder_paths.cache_dir):
    INSIGHTFACE_DIR = os.path.join(folder_paths.cache_dir, "insightface")

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None, in_settings=None):
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda', model_rootpath="/stable-diffusion-cache/models/facexlib/weights")
    face_emb = arcface_model(arc_face_image)[0] # [512], normalized
    return face_emb


class InfiniteYou(torch.nn.Module):
    def __init__(self, adapter_model):
        super().__init__()
        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(adapter_model["image_proj"])

        # Load face encoder
        self.app_640 = FaceAnalysis(name='antelopev2', 
                                root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        self.app_320 = FaceAnalysis(name='antelopev2', 
                                root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        self.app_160 = FaceAnalysis(name='antelopev2', 
                                root=INSIGHTFACE_DIR, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        self.arcface_model = init_recognition_model('arcface', device='cuda', model_rootpath="/stable-diffusion-cache/models/facexlib/weights")
        

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=8,
            embedding_dim=512,
            output_dim=4096,
            ff_mult=4
        )
        return image_proj_model
    
    def detect_face(self, id_image_cv2):
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def get_face_embed_and_landmark(self, ref_image):
        id_image_cv2 = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
        face_info = self.detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        landmark = face_info['kps']
        id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float().cuda()
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        return id_embed, face_info['kps']

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds



def add_noise(image, factor):
    seed = int(torch.sum(image).item()) % 1000000007
    torch.manual_seed(seed)
    mask = (torch.rand_like(image) < factor).float()
    noise = torch.rand_like(image)
    noise = torch.zeros_like(image) * (1-mask) + noise * mask

    return factor*noise
class ApplyInfiniteYou:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "ref_image": ("IMAGE", ),
                "latent_image": ("LATENT", ),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"), ),
                "weight": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "vae": ("VAE", ),
                "fixed_face_pose": ("BOOLEAN", {"default": False, "tooltip": "Fix the face pose from reference image."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_infinite_you"
    CATEGORY = "ComfyUI-InfiniteYou"
    
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")

        model = InfiniteYou(
            adapter_model_state_dict
        )

        return model

    def apply_infinite_you(self, adapter_file, control_net, ref_image, model, positive, negative, start_at, end_at, vae, latent_image, fixed_face_pose, weight=.99,  ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
        ref_image = tensor_to_image(ref_image)
        ref_image = PIL.Image.fromarray(ref_image.astype(np.uint8))
        ref_image = ref_image.convert("RGB")

        #load resampler model
        infinteyou_model = self.load_model(adapter_file)

        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        cn_strength = weight if cn_strength is None else cn_strength

        #get face embedding
        face_embed, landmark = infinteyou_model.get_face_embed_and_landmark(ref_image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')
        

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        #apply infinite you
        infinteyou_model = infinteyou_model.to(self.device, dtype=self.dtype)
        image_prompt_embeds, uncond_image_prompt_embeds = infinteyou_model.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        #get face kps
        out = []
        height = latent_image['samples'].shape[2] * 8
        width = latent_image['samples'].shape[3] * 8
        if fixed_face_pose:
            control_image = resize_and_pad_image(ref_image, (width, height))
            image_kps = draw_kps(control_image, landmark)
        else:
            image_kps = np.zeros([height, width, 3])
            image_kps = PIL.Image.fromarray(image_kps.astype(np.uint8))

        out.append(image_kps)
        out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        face_kps = out
        

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at), vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(model, cond_uncond[0], cond_uncond[1], latent_image)
    

class FaceCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "ref_image_1": ("IMAGE", ),
                "ref_image_2": ("IMAGE", ),
                "latent_image": ("LATENT", ),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"), ),
                "weight": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "vae": ("VAE", ),
                "fixed_face_pose": ("BOOLEAN", {"default": False, "tooltip": "Fix the face pose from reference image."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_face_combine"
    CATEGORY = "ComfyUI-InfiniteYou"
    
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")

        model = InfiniteYou(
            adapter_model_state_dict
        )

        return model

    def apply_face_combine(self, adapter_file, control_net, ref_image_1, ref_image_2, model, positive, negative, start_at, end_at, vae, latent_image, fixed_face_pose, balance = 0.5, weight=.99,  ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
        ref_image_1 = tensor_to_image(ref_image_1)
        ref_image_1 = PIL.Image.fromarray(ref_image_1.astype(np.uint8))
        ref_image_1 = ref_image_1.convert("RGB")

        ref_image_2 = tensor_to_image(ref_image_2)
        ref_image_2 = PIL.Image.fromarray(ref_image_2.astype(np.uint8))
        ref_image_2 = ref_image_2.convert("RGB")

        #load resampler model
        infinteyou_model = self.load_model(adapter_file)


        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        cn_strength = weight if cn_strength is None else cn_strength

        #get face embedding
        face_embed_1, landmark_1 = infinteyou_model.get_face_embed_and_landmark(ref_image_1)
        if face_embed_1 is None:
            raise Exception('Reference Image: No face detected.')

        face_embed_2, landmark_2 = infinteyou_model.get_face_embed_and_landmark(ref_image_2)
        if face_embed_2 is None:
            raise Exception('Reference Image: No face detected.')

        #copied from https://github.com/vuongminh1907/ComfyUI_ZenID/blob/main/ZenID.py#L596
        face_embed = face_embed_1 * balance + face_embed_2 * (1 - balance)

        clip_embed = face_embed

        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        #apply infinite you
        infinteyou_model = infinteyou_model.to(self.device, dtype=self.dtype)
        image_prompt_embeds, uncond_image_prompt_embeds = infinteyou_model.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        #get face kps
        out = []
        height = latent_image['samples'].shape[2] * 8
        width = latent_image['samples'].shape[3] * 8
        if fixed_face_pose:
            control_image = resize_and_pad_image(ref_image_1, (width, height))
            image_kps = draw_kps(control_image, landmark_1)
        else:
            image_kps = np.zeros([height, width, 3])
            image_kps = PIL.Image.fromarray(image_kps.astype(np.uint8))

        out.append(image_kps)
        out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        face_kps = out
        

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at), vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(model, cond_uncond[0], cond_uncond[1], latent_image)

class SwapFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "ref_image": ("IMAGE", ),
                "image": ("IMAGE", ),
                "adapter_file": (folder_paths.get_filename_list("InfiniteYou"), ),
                "weight": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "vae": ("VAE", ),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_face_swap"
    CATEGORY = "ComfyUI-InfiniteYou"

    def apply_face_swap(self, adapter_file, control_net, model, ref_image, image, clip, start_at, end_at, vae, weight=.99,  ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
        ref_image = tensor_to_image(ref_image)
        ref_image = PIL.Image.fromarray(ref_image.astype(np.uint8))
        ref_image = ref_image.convert("RGB")

        tensor_image = image.clone()
        image = tensor_to_image(image)
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image = image.convert("RGB")


        #load resampler model
        infinteyou_model = self.load_model(adapter_file)

        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        cn_strength = weight if cn_strength is None else cn_strength

        if mask is None:
            mask, landmark = self.prepare_mask_and_landmark(infinteyou_model, image, 9)
        else:
            _, landmark = infinteyou_model.get_face_embed_and_landmark(image)

        #prepare latent
        prompt = " "
        neg_prompt = "ugly, blurry"
        positive = self.encode_prompt(clip, prompt)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": float(1.5)})
        negative = self.encode_prompt(clip, neg_prompt)

        tensor_image = tensor_image.to(self.device, dtype=self.dtype)
        positive, negative, latent_image = self.prepare_condition_inpainting(positive, negative, tensor_image, vae, mask)

        #get face embedding
        face_embed, _ = infinteyou_model.get_face_embed_and_landmark(ref_image)

        if face_embed is None:
            raise Exception('Reference Image: No face detected.')
        
        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        #apply infinite you
        infinteyou_model = infinteyou_model.to(self.device, dtype=self.dtype)
        image_prompt_embeds, uncond_image_prompt_embeds = infinteyou_model.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        #get face kps
        out = []
        control_image = image
        image_kps = draw_kps(control_image, landmark)

        out.append(image_kps)
        out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        face_kps = out
        

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at), vae=vae)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net
                
                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False
        
        

        return(model, cond_uncond[0], cond_uncond[1], latent_image)

        
    def load_model(self, adapter_file):
        ckpt_path = folder_paths.get_full_path("InfiniteYou", adapter_file)
        adapter_model_state_dict = torch.load(ckpt_path, map_location="cpu")

        model = InfiniteYou(
            adapter_model_state_dict
        )

        return model

    def encode_prompt(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    def prepare_condition_inpainting(self, positive, negative, pixels, vae, mask):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1).to(self.device, dtype=self.dtype)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return out[0], out[1], out_latent

    def prepare_mask_and_landmark(self, model, image, blur_kernel):
        image_detect = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_info = model.detect_face(image_detect)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        landmark = face_info['kps']

        mask = np.zeros((image.size[1], image.size[0]))
        x1, y1, x2, y2 = face_info['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the expansion amount (1/3 of each dimension)
        width = x2 - x1
        height = y2 - y1
        expand_x = width // 3
        expand_y = height // 3

        # Expand the bounding box
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = min(image.size[0], x2 + expand_x)
        new_y2 = min(image.size[1], y2 + expand_y)

        # Apply the expanded mask
        mask[new_y1:new_y2, new_x1:new_x2] = 1

        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return mask, landmark


NODE_CLASS_MAPPINGS = {
    "FaceSwap_InfiniteYou": SwapFace,
    "InfiniteYouApply": ApplyInfiniteYou,   
    "FaceCombine": FaceCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteYouApply": "InfiniteYou Apply",
    "FaceCombine": "Face Combine (InfiniteYou)",
    "FaceSwap_InfiniteYou": "Face Swap (InfiniteYou)",
}