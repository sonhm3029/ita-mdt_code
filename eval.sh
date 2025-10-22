CUDA_VISIBLE_DEVICES=0 python scr/eval.py \
    --vitonhd_dataroot='DATA/zalando-hd-resized' \
    --batch_size=16 \
    --workers=8 \
    --gen_folder='results/ita-mdt_weights_ema_0.9999_2000000/VITON-HD/pair' \
    --dataset="vitonhd" \
    --test_order="paired" \
    --category="upper_body" 