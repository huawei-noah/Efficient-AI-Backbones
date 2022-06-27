CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1  main.py --eval \
--cfg configs/snnmlp_tiny_patch4_lif4_224.yaml --resume checkpoints/snnmlp_tiny_patch4_lif4_224.pth --data-path /imagenet  --cache-mode no --lif 4
