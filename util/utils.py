import copy
import torch
import torch.nn.functional as F

import time

def point_tracking(F0, F1, handle_points, handle_points_init, args):
    start_time = time.time()
    max_duration = 300
    print("start point tracking")
    with torch.no_grad():
        batch_size = F0.shape[0]
        _, max_r, max_c = F0.shape[1:]  # 각 이미지의 차원
        for b in range(batch_size):  # 배치에 대한 반복
            if time.time() - start_time > max_duration:
                assert 0, "Max duration exceeded, stopping early."
            for i in range(len(handle_points[b])):  # 핸들 포인트에 대한 반복
                pi0, pi = handle_points_init[b][i], handle_points[b][i]
                f0 = F0[b, :, int(pi0[0]), int(pi0[1])].unsqueeze(dim=0)

                r1, r2 = max(0, int(pi[0])-args.r_p), min(max_r, int(pi[0])+args.r_p+1)
                c1, c2 = max(0, int(pi[1])-args.r_p), min(max_c, int(pi[1])+args.r_p+1)
                F1_neighbor = F1[b, :, r1:r2, c1:c2].unsqueeze(dim=0)
                all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
                all_dist = all_dist.squeeze(dim=0)
                row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])

                handle_points[b][i][0] = r1 + row
                handle_points[b][i][1] = c1 + col
    print("point tracking time: ", time.time() - start_time)
    return handle_points
    
# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat,
                              y1,
                              y2,
                              x1,
                              x2):
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor : y1_floor+dy, x1_floor : x1_floor+dx]
    Ib = feat[:, :, y1_cell : y1_cell+dy, x1_floor : x1_floor+dx]
    Ic = feat[:, :, y1_floor : y1_floor+dy, x1_cell : x1_cell+dx]
    Id = feat[:, :, y1_cell : y1_cell+dy, x1_cell : x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd
    
def check_handle_reach_target(handle_points,
                              target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p,q: (p-q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


def diffusion_update(model,
                     init_code,
                     vae_latent,
                     image_embeddings,
                     t,
                     handle_points,
                     target_points,
                     mask,
                     args):
    
    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    
    do_classifier_free_guidance = True if args.guidance_scale > 1 else False
    added_time_ids = model._get_add_time_ids(
            fps=args.num_frames-1,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            dtype=torch.float32,
            batch_size=1,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
    
    vae_latent = vae_latent.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)
    
    # the init output feature of unet
    with torch.no_grad():
        latent_model_input = torch.cat([init_code] * 2) if do_classifier_free_guidance == True else init_code
        latent_model_input = torch.cat([latent_model_input, vae_latent], dim=2)
        unet_output, F0 = model.forward_unet_features(latent_model_input,
                                                      t,
                                                      encoder_hidden_states=image_embeddings,
                                                      added_time_ids=added_time_ids,
                                                      layer_idx=args.unet_feature_idx,
                                                      interp_res_h=args.sup_res_h,
                                                      interp_res_w=args.sup_res_w,)
        x_prev_0,_ = model.step(unet_output, t, init_code)
        
    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)
    
    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[3],init_code.shape[4]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0
    
    # TODO: prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            latent_model_input = torch.cat([init_code] * 2) if do_classifier_free_guidance == True else init_code
            latent_model_input = torch.cat([latent_model_input, vae_latent], dim=2)
            unet_output, F1 = model.forward_unet_features(latent_model_input,
                                                    t,
                                                    encoder_hidden_states=image_embeddings,
                                                    added_time_ids=added_time_ids,
                                                    layer_idx=args.unet_feature_idx,
                                                    interp_res_h=args.sup_res_h,
                                                    interp_res_w=args.sup_res_w,)
            x_prev_updated,_ = model.step(unet_output, t, init_code)
            
            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('init handle points', handle_points_init)
                print('new handle points', handle_points)
            # break if all handle points have reached the targets
            # if check_handle_reach_target(handle_points, target_points):
            #     break
            
            loss = 0.0
            batch_size, _, max_r, max_c = F0.shape
            for b in range(batch_size):
                for i in range(len(handle_points[b])):
                    print(f"b={b}, i={i}")
                    pi, ti = handle_points[b][i], target_points[b][i]
                    # skip if the distance between target and source is less than 1
                    if (ti - pi).norm() < 2.:
                        continue
                    
                    di = (ti - pi) / (ti - pi).norm() 
                    
                    # motion supervision
                    # with boundary protection
                    r1, r2 = max(0,int(pi[0])-args.r_m), min(max_r,int(pi[0])+args.r_m+1)
                    c1, c2 = max(0,int(pi[1])-args.r_m), min(max_c,int(pi[1])+args.r_m+1)
                    F1_patch = F1[b].unsqueeze(0)
                    f0_patch = F1_patch[:, :, r1:r2, c1:c2].detach()
                    f1_patch = interpolate_feature_patch(F1_patch,r1+di[0],r2+di[0],c1+di[1],c2+di[1])
                    # original code, without boundary protection
                    # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                    # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                    loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)
                    
                # masked region must stay unchanged
                if using_mask:
                    x_prev_updated_patch = x_prev_updated[:, b].unsqueeze(1)
                    x_prev_0_patch = x_prev_0[:, b].unsqueeze(1)
                    loss += args.lam * ((x_prev_updated_patch-x_prev_0_patch)*(1.0-interp_mask)).abs().sum()
                # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
                print('loss total=%f'%(loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code