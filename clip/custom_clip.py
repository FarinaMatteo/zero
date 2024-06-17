
import torch
import torch.nn as nn
from contextlib import nullcontext

from clip import load, tokenize
from data.imagenet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

from maple_clip.maple import CustomCLIP as MaPLeCLIP


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        """
        This code is identical to the `forward` function of OpenAI's CLIP, but it does not 
        tokenize text on the fly to then lookup the word embedding table. 
        In constrast, the passed `prompts` are optimized / learned by `PromptLearner`.
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class ClipWrapper(nn.Module):
    def __init__(self, 
                 device, 
                 arch="ViT-B/16",
                 ctx_init=None, 
                 freeze_vision=False,
                 freeze_text=False,
                 cache_text_features=False,
                 use_text_templates=False):

        super(ClipWrapper, self).__init__()
        self.arch = arch
        clip, _, _ = load(arch, device=device)
        
        self.name = f"CLIP-{arch}"
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        
        self.cache_text_features = cache_text_features
        self.use_text_templates = use_text_templates

        if self.use_text_templates:
            from data.imagenet_prompts import tip_imagenet_templates
            self.prompt_templates = tip_imagenet_templates
            self.token_embedding = clip.token_embedding
        else:
            self.hard_prompt = ctx_init.replace("_", " ")
            self.token_embedding = clip.token_embedding
        

        self.freeze_text = freeze_text
        self.freeze_vision = freeze_vision
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    @property
    def device(self):
        return self.image_encoder.conv1.weight.device
    
    @property
    def temp(self):
        return 1/self.logit_scale.exp()
        

    def reset_classnames(self, classnames, arch=None):
        self.classnames = classnames
        if arch is None:
            arch = self.arch
        
        if self.use_text_templates:
            hard_prompts = []
            for cls in classnames:
                for template in self.prompt_templates:
                    hard_prompts.append(template.format(cls))
            self.tokenized_descriptions = tokenize(hard_prompts).to(self.device)
        else:
            class_descriptions = [f"{self.hard_prompt} {name}." for name in classnames]
            self.tokenized_descriptions = tokenize(class_descriptions).to(self.device)
        
        if self.cache_text_features and hasattr(self, "text_features"):
            delattr(self, "text_features")

    def get_text_features(self):

        tokenized_prompts = self.tokenized_descriptions
        prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        
        t_features = self.text_encoder(prompts, tokenized_prompts)
        norm_feats = t_features / t_features.norm(dim=-1, keepdim=True)

        if self.use_text_templates:
            norm_feats = norm_feats.chunk(len(self.classnames), dim=0)
            classifiers = []
            for i, nf in enumerate(norm_feats):
                classifier = nf.mean(dim=0)
                classifier /= classifier.norm()
                classifiers.append(classifier)
            norm_feats = torch.stack(classifiers, dim=0)

        return norm_feats  
    

    def get_image_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


    def inference(self, image):
        # encode images (you don't need gradients if you only learn the textual prompts)
        cm = torch.no_grad if self.freeze_vision else nullcontext
        with cm():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # encode texts
        cm = torch.no_grad if self.freeze_text  else nullcontext
        with cm():
            if self.cache_text_features:
                if not hasattr(self, "text_features") or self.text_features is None:
                    self.text_features = self.get_text_features()
                text_features = self.text_features
            else:
                text_features = self.get_text_features()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def forward(self, image):
        return self.inference(image)


def get_clip(clip_arch, device, ctx_init, 
             freeze_text=False, freeze_vision=False, 
             cache_text_features=False, use_text_templates=False, 
             maple_weights=None, n_maple_ctx=2, maple_prompt_depth=3, maple_seed=1):

    if not maple_weights:
        return ClipWrapper(
            device, 
            arch=clip_arch,
            ctx_init=ctx_init, 
            freeze_text=freeze_text,
            freeze_vision=freeze_vision,
            cache_text_features=cache_text_features,
            use_text_templates=use_text_templates,
        )
    else:
        model = MaPLeCLIP(
            clip_arch=clip_arch,
            n_ctx=n_maple_ctx,
            ctx_init=ctx_init,
            prompt_depth=maple_prompt_depth,
            freeze_text=freeze_text,
            freeze_vision=freeze_vision,
            cache_text_features=cache_text_features
        )
        path_to_pretraining = f"weights/maple_seed{maple_seed}.pth"
        model.load_pretrained(ckpt_path=path_to_pretraining)
        return model

