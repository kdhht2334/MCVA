python train.py --gpu-id 0 \
                --loss PPGML_PA \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --lr 1e-4 \
				--epochs 70 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 10 \
                --lr-decay-gamma 0.5 \
#
# Possible options (4): PPGML_Triplet, PPGML_Contrastive, PPGML_PA, PPGML_MS
# PPGML-PA:       lr (1e-4) / bs: 180 (BN), 120 (R50)
# PPGML-MS:       lr (3e-5) / bs: 180 (BN), 120 (R50)
# PPGML-Tri/Cont: lr (1e-5) / bs: 112 (all)
#
# directory for tuning: anaconda3/envs/pytorch/lib/python3.6/site-packages/pytorch_metric_learning
