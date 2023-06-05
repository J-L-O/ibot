Here we document the commands used to finetune the ImageNet-pretrained DINO model on other datasets.

CUB

```bash
    python run_with_submitit.py \
    --act_in_head gelu \
    --arch vit_base \
    --clip_grad 0.3 \
    --drop_path 0.1 \
    --epochs 400 \
    --saveckp_freq 200 \
    --freeze_last_layer 3 \
    --global_crops_number 2 \
    --global_crops_scale 0.32 1.0 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --local_crops_number 10 \
    --local_crops_scale 0.05 0.32 \
    --lr 0.000125 \
    --min_lr 2e-6 \
    --momentum_teacher 0.996 \
    --norm_last_layer true \
    --optimizer adamw \
    --out_dim 8192 \
    --patch_out_dim 8192 \
    --patch_size 16 \
    --pred_ratio 0.0 0.7 \
    --pred_ratio_var 0.0 0.05 \
    --pred_shape rand \
    --pred_start_epoch 0 \
    --seed 0 \
    --shared_head true \
    --shared_head_teacher true \
    --teacher_patch_temp 0.07 \
    --teacher_temp 0.07 \
    --use_fp16 true \
    --use_masked_im_modeling true \
    --warmup_epochs 10 \
    --warmup_teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 50 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --window_size 7 \
    --grad_from_block 11 \
    --data_path /hpi/fs00/share/fg-meinel/datasets/GCD-datasets/cub/ \
    --output_dir /hpi/fs00/home/jona.otholt/ibot/finetuning/CUB200/ \
    --load_from pretrained.pth \
    --batch_size_per_gpu 256 \
    --ngpus 4 \
    --nodes 1 \
    --account meinel-mlai \
    --partition sorcery \
    --cpus_per_task 20 \
    --constraint 'ARCH:X86'
```
