
torchrun --nproc_per_node 8 -m \
    --master_addr=127.0.0.3 --master_port=29533 \
    training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="path/to/cc3m_train.csv,path/to/cc12m.csv"  \
    --val-data="path/to/flickrir/"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --data-root path/to/cc3m/images/,path/to/cc12m/images/ \
    --val-data-root path/to/flickrir \
    --imagenet-val=path/to/imagenet/val/ \
    --imagenet-v2=path/to/ImageNetV2-matched-frequency/ \
    --imagenet-r=path/to/imagenet-rendition/imagenet-r/ \
    --imagenet-a=path/to/imagenet-a/imagenet-a/ \
    --imagenet-sketch=path/to/imagenet-sketch/sketch/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=16 \
    --model ViT-T-16 \
    --resume path/to/model.pt \
    --logs path/to/logs/  \
    --eval \
    --tag eval_flickr