import copy
import torch
import torch.nn as nn

from torch.nn import functional as F
from maple_clip import clip
from maple_clip import load, tokenize
from maple_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(arch, n_ctx):
    backbone_name = arch
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": n_ctx}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        try:
            self.dtype = clip_model.dtype
        except:
            self.dtype = clip_model.text_projection.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt  
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx, ctx_init, prompt_depth):
        super().__init__()
        try:
            dtype = clip_model.dtype
        except:
            dtype = clip_model.visual.conv1.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Default is 1, which is compound shallow prompting
        assert prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = prompt_depth  # max=12, but will create 11 such shared prompts

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.to(clip_model.device)).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        self.n_ctx = n_ctx
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {self.n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]


        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts.to(clip_model.device)
            ).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIPWithMaple(nn.Module):
    def __init__(self, device, classnames, arch="ViT-B/16", n_ctx=2, ctx_init="a photo of a", prompt_depth=3, load_from=None):
        super().__init__()
        
        # initialize the base clip model either from openai's clip or from open_clip
        self.arch = arch
        clip_model = load_clip_to_cpu(arch, n_ctx)
        clip_model = clip_model.to(device)
        setattr(clip_model, "device", device)
        
        # initialize the prompt learner
        self.prompt_learner = MultiModalPromptLearner(
            clip_model,
            classnames,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            prompt_depth=prompt_depth
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # initialize the encoders
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        # temperature (log) and datatype
        self.logit_scale = clip_model.logit_scale
        self.dtype = self.text_encoder.dtype

        if load_from is not None:
            self.load_from = load_from
            self.load_pretrained_maple(load_from, device)

        del clip_model
    
    def load_pretrained_maple(self, load_from, device):
        ckpt = torch.load(load_from, map_location=torch.device(device))
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # Ignore fixed token vectors
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        msg = self.load_state_dict(state_dict=state_dict, strict=False)

        print('='*20)
        for k in msg.missing_keys:
            print(f"Missing key: {k}")
        
        print('='*20)
        for k in msg.unexpected_keys:
            print(f"Unexpected key: {k}")


    def reset_classnames(self, classnames, arch):
        clip_model = load_clip_to_cpu(arch, self.prompt_learner.n_ctx)
        clip_model = clip_model.to(self.device)
        setattr(clip_model, "device", self.device)
        self.prompt_learner = MultiModalPromptLearner(
            clip_model,
            classnames,
            n_ctx=self.prompt_learner.n_ctx,
            ctx_init=self.prompt_learner.prompt_prefix,
            prompt_depth=self.prompt_learner.compound_prompts_depth
        ).to(self.device)
        self.load_pretrained_maple(self.load_from, self.device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        del clip_model


    @property
    def device(self):
        return self.image_encoder.conv1.weight.device


    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])