from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
from masked_diffusion import create_diffusion
# import models_mae

# def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
#     # build model
#     model = getattr(models_mae, arch)()
#     # load model
#     checkpoint = torch.load(chkpt_dir, map_location='cpu')
#     model.load_state_dict(checkpoint['model'], strict=False)
#     return model

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, config, cls_fn=None, classes=None):
    with torch.no_grad():
        # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        # if config.model.type == 'mdt':
        #     vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(x.device)

        #setup vectors used in the algorithm
        singulars = H_funcs.singulars()
        Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
        Sigma[:singulars.shape[0]] = singulars
        U_t_y = H_funcs.Ut(y_0)
        Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

        #initialize x_T as given in the paper
        largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
        largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
        large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
        inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
        inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
        inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

        # implement p(x_T | x_0, y) as given in the paper
        # if eigenvalue is too small, we just treat it as zero (only for init) 
        init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
        init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
        init_y = init_y.view(*x.size())
        remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
        remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
        init_y = init_y + remaining_s * x
        init_y = init_y / largest_sigmas

        #setup iteration variables
        x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        tvu.save_image(init_y, os.path.join(config.image_folder, 'init_y.png'))
        tvu.save_image(x, os.path.join(config.image_folder, 'init_x.png'))
        # print(f"x.shape: {x.shape}")
        
        if config.model.type == 'mdt':
            diffusion = create_diffusion('', # str(config.diffusion.num_diffusion_timesteps),
                                         diffusion_steps=config.diffusion.num_diffusion_timesteps,
                                         beta_start=config.diffusion.beta_start,
                                         beta_end=config.diffusion.beta_end)

        #iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq[:100]), reversed(seq_next[:100])), desc='efficient_generalized_steps'):
            t = (torch.ones(n, dtype=int) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            classes = classes.to(x.device)
            if cls_fn == None:
                if config.model.type == 'mdt':
                    # latent = torch.cat([xt, xt], dim=0).to(x.device)
                    # print(f"latent.shape: {latent.shape}")
                    # tmp_t = torch.cat([t, t], dim=0).to(x.device)
                    # y = classes.clone()
                    # y_null = torch.full_like(y, 1000)
                    # y = torch.cat([y, y_null], dim=0)
                    
                    # model_kwargs = dict(y=classes, diffusion_steps=1000, cfg_scale=5.0, scale_pow=0.01)
                    model_kwargs = dict(y=classes)
                    et = diffusion.p_sample(
                      model,
                      xt,
                      t,
                      clip_denoised=False,
                      model_kwargs=model_kwargs,
                    )['sample']
                    # et = model(xt, t, classes)
                    # print(f"et.shape: {et.shape}")
                    # et = model(xt, t, classes_all)
                    # et, _ = et.chunk(2, dim=1)  # Remove null class samples for cfg
                    # print(f"et.shape: {et.shape}")
                    # et = vae.decode(samples / 0.18215).sample
                else:
                    et = model(xt, t)
                #tvu.save_image(et, f"samples/sample_{i}.jpg")
                # et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)

            if et.size(1) == 6:
                et = et[:, :3]
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            #variational inference conditioned on y
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            xt_mod = xt / at.sqrt()[0, 0, 0, 0]
            V_t_x = H_funcs.Vt(xt_mod)
            SVt_x = (V_t_x * Sigma)[:, :U_t_y.shape[1]]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)

            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)
            
            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            #missing pixels
            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])
            
            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all 3 cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))


    return xs, x0_preds
