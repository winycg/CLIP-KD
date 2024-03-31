conda activate ycg_clip_kd
CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun --nproc_per_node 4 -m \
    --master_addr=127.0.0.3 --master_port=29533 \
    training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/data/dataset/Conceptual_Captions/cc3m_train.csv,/data/dataset/cc12m/cc12m.csv"  \
    --val-data="/data/dataset/retrieval/cocoir/"  \
    --data-root /data/dataset/,/data/dataset/ \
    --val-data-root /data/dataset/retrieval/cocoir/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/data/winycg/dataset/ImageNet/val \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=16 \
    --model RN50 \
    --resume /data/winycg/clip_models_zoo/laion/2024_02_27-17_43_47-model_RN50-lr_0.001-b_256-epochs_32-tag_cc3m-cc12m-baseline-RN50/checkpoints/epoch_32.pt \
    --logs /data/winycg/checkpoints/clip_kd_checkpoints/2024_2_eval_results/ \
    --eval \
    --tag eval_coco



conda activate ycg_clip_kd
CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun --nproc_per_node 4 -m \
    --master_addr=127.0.0.3 --master_port=29533 \
    training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/data/dataset/Conceptual_Captions/cc3m_train.csv,/data/dataset/cc12m/cc12m.csv"  \
    --val-data="/data/dataset/retrieval/flickrir"  \
    --data-root /data/dataset/,/data/dataset/ \
    --val-data-root /data/dataset/retrieval/flickrir \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/data/winycg/dataset/ImageNet/val \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=16 \
    --model RN50 \
    --resume /data/winycg/clip_models_zoo/laion/2024_02_27-17_43_47-model_RN50-lr_0.001-b_256-epochs_32-tag_cc3m-cc12m-baseline-RN50/checkpoints/epoch_32.pt \
    --logs /data/winycg/checkpoints/clip_kd_checkpoints/2024_2_eval_results/ \
    --eval \
    --tag eval_flickrir