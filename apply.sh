 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/InternVideo2/multi_modality/scripts/spfs/clip/B14/config.py b/InternVideo2/multi_modality/scripts/spfs/clip/B14/config.py
index 0fcea883c6c7de9f6b90772b5d51f6b47e377e6e..bf5fc1f12c1c1a0dba1e81f82f11197aced72527 100644
--- a/InternVideo2/multi_modality/scripts/spfs/clip/B14/config.py
+++ b/InternVideo2/multi_modality/scripts/spfs/clip/B14/config.py
@@ -108,50 +108,51 @@ optimizer = dict(
     weight_decay=0.01,
     max_grad_norm=0.7,
     # use a different lr for some modules, e.g., larger lr for new modules
     different_lr=dict(enable=False, module_names=["streaming_vision_encoder.vit_lite"], lr=2e-6),
 )
 
 scheduler = dict(sched='cosine', epochs=2, min_lr_multi=0.01, warmup_epochs=0.1)
 
 evaluate = False
 deep_fusion = False
 evaluation = dict(
     eval_frame_ensemble="concat",  # [concat, max, mean, lse]
     eval_x_only=False,
     k_test=128,
     eval_offload=True,  # offload gpu tensors to cpu to save memory.
 )
 
 use_half_precision = True
 use_bf16 = True
 gradient_checkpointing = True
 
 # =================== spfs hyperparams =====================
 
 lambda_calib = 1.0
 lambda_skip = 0.1
+calibration_loss_fn = 'bce'
 
 # ========================= wandb ==========================
 wandb = dict(
     enable=True,
     entity="qingy2019-conker-mobile-inc-",
     project="window_iv2",
 )
 dist_url = "env://"
 device = "cuda"
 mode = "pt"
 
 # ========================= others ==========================
 output_dir = './train_outputs_spfs/'
 resume = False
 debug = False
 log_freq = 1
 seed = 42
 
 save_latest = False
 save_iter = 100
 
 auto_resume = False
 pretrained_path = ""
 
 deepspeed = dict(
diff --git a/InternVideo2/multi_modality/tasks_clip/train_spfs.py b/InternVideo2/multi_modality/tasks_clip/train_spfs.py
index 765de930d0582bbcf737e9ad37d8cbdfeb56c071..d251e38cdb7c030f9207ac71d24d18f72e859b0d 100644
--- a/InternVideo2/multi_modality/tasks_clip/train_spfs.py
+++ b/InternVideo2/multi_modality/tasks_clip/train_spfs.py
@@ -106,52 +106,59 @@ def train(
 
     media_types = get_media_types(train_loaders)
     if config.distributed:
         for loader in train_loaders:
             loader.sampler.set_epoch(epoch)
 
     seed = config.seed + epoch
     train_loader_agg = MetaLoader_rs(
         name2loader=dict(zip(media_types, train_loaders)),
         skip_num=skip_num,
         seed=seed,
     )
 
     progress_bar = tqdm(
         train_loader_agg,
         total=len(train_loader_agg),
         desc=f"Training: [Epoch {epoch}]",
         disable=not is_main_process(),
     )
 
     # Loss fns and hyper-param
     cosine_loss_fn = lambda pred, target: 1 - torch.nn.functional.cosine_similarity(
         pred, target.detach(), dim=-1
     ).mean()
     bce_loss_fn = BCEWithLogitsLoss()
+    mse_loss_fn = MSELoss()
     primary_loss_fn = MSELoss()
 
+    calibration_loss_type = config.get("calibration_loss_fn", "bce").lower()
+    if calibration_loss_type not in ["bce", "mse"]:
+        raise ValueError(
+            f"Unsupported calibration_loss_fn {calibration_loss_type}. Choose 'bce' or 'mse'."
+        )
+
     lambda_calib = config.get("lambda_calib", 1.0)
     lambda_skip = config.get("lambda_skip", 0.1)
     MODEL_MAX_FRAMES = config.num_frames
 
     for i, data_pair in enumerate(progress_bar):
         batch = data_pair[1]
         if len(batch) == 4:
             image, _, _, teacher_emb = batch
         else:
             image, _, _ = batch
             teacher_emb = None
 
         image = image.to(device, non_blocking=True)
         image = image.permute(0, 2, 1, 3, 4)
 
         B, C, T, H, W = image.shape
         assert T >= MODEL_MAX_FRAMES, f"Video batch contains sequences shorter than {MODEL_MAX_FRAMES} frames."
 
         with torch.amp.autocast('cuda', enabled=config.use_half_precision, dtype=data_type):
             # Warm up hidden state
             h = model.streaming_vision_encoder.init_hidden(batch_size=B, device=device)
 
             for t in range(MODEL_MAX_FRAMES - 1):
                 frame = image[:, :, t, :, :].unsqueeze(2) # [B, C, 1, H, W]
                 with torch.no_grad():
@@ -186,54 +193,61 @@ def train(
                             window_start = idx_curr - MODEL_MAX_FRAMES + 1
                             window_end = idx_curr + 1
                             curr_window = image[:, :, window_start:window_end, :, :]
                             target_curr = model_without_ddp.vision_align(
                                 model_without_ddp.vision_encoder(curr_window)
                             )
                     else:
                         target_curr = None
 
                 # ----------
 
                 mu_t, logvar = model.streaming_vision_encoder.rnn.predict_next_feat()
                 conf_logit = -logvar.squeeze(-1)
 
                 L_pred = cosine_loss_fn(mu_t, target_next)
 
                 if epoch == 0:
                     # Phase-1: only train predictor head (primary & other losses = 0)
                     loss = L_pred
                     L_primary = L_calib = L_skip = torch.tensor(0.0, device=device) # Set others to 0
                 else:
                     # Phase-2: joint fine-tuning
                     L_primary = primary_loss_fn(out_t, target_curr)
 
                     with torch.no_grad():
-                        target_c = (
-                            torch.nn.functional.cosine_similarity(mu_t, target_next, dim=-1) >= 0.85
-                        ).float()
-                    L_calib = bce_loss_fn(conf_logit.squeeze(-1), target_c)
+                        sim_score = torch.nn.functional.cosine_similarity(
+                            mu_t, target_next, dim=-1
+                        )
+
+                    if calibration_loss_type == "bce":
+                        target_c = (sim_score >= 0.85).float()
+                        L_calib = bce_loss_fn(conf_logit.squeeze(-1), target_c)
+                    else:  # mse
+                        L_calib = mse_loss_fn(
+                            torch.sigmoid(conf_logit.squeeze(-1)), sim_score
+                        )
                     L_skip = -(torch.log(torch.sigmoid(conf_logit) + 1e-8)).mean()
 
                     loss = L_primary + L_pred + lambda_calib * L_calib + lambda_skip * L_skip
 
                 if hasattr(config, "deepspeed") and config.deepspeed.enable:
                     model.backward(loss)
                     model.step()
                 else:  # standard AMP path
                     optimizer.zero_grad()
                     if config.use_half_precision:
                         scaler.scale(loss).backward()
                         if config.optimizer.max_grad_norm > 0:
                             scaler.unscale_(optimizer)
                             torch.nn.utils.clip_grad_norm_(
                                 model.parameters(), config.optimizer.max_grad_norm
                             )
                         scaler.step(optimizer)
                         scaler.update()
                     else:
                         loss.backward()
                         if config.optimizer.max_grad_norm > 0:
                             torch.nn.utils.clip_grad_norm_(
                                 model.parameters(), config.optimizer.max_grad_norm
                             )
                         optimizer.step()
 
EOF
)
