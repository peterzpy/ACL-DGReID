#logs/
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/train_net.py --config-file ./configs/bagtricks_DR50_mix.yml --num-gpus 4