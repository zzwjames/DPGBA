
models=(GCN)

# python -u pre.py \
#     --prune_thr=0.05\
#     --target_class=2\
#     --dataset=Cora\
#     --k=50\
#     --homo_loss_weight=50\
#     --vs_number=10\
#     --hidden=32\
#     --train_lr=0.005\
#     --lr=0.005\
#     --test_model=GCN\
#     --homo_boost_thrd=0.5\
#     --weight_targetclass=10\
#     --weight_ood=50\
#     --defense_mode=reconstruct\
#     --evaluate_mode=overall\
#     --epochs=401\
#     --trigger_size=3\
#     --trojan_epochs=201


## Cora ##
# python -u run_adaptive.py \
#     --prune_thr=0.05\
#     --target_class=2\
#     --dataset=Cora\
#     --k=50\
#     --homo_loss_weight=50\
#     --vs_number=10\
#     --hidden=32\
#     --train_lr=0.005\
#     --lr=0.005\
#     --test_model=GCN\
#     --homo_boost_thrd=0.5\
#     --weight_targetclass=10\
#     --weight_ood=50\
#     --evaluate_mode=overall\
#     --epochs=401\
#     --trigger_size=3\
#     --trojan_epochs=201


## PubMed ##
# python -u run_adaptive.py \
#     --prune_thr=0.2\
#     --dataset=Pubmed\
#     --hidden=64\
#     --homo_loss_weight=160\
#     --vs_number=40\
#     --test_model=GCN\
#     --homo_boost_thrd=0.5\
#     --train_lr=0.002\
#     --lr=0.002\
#     --target_class=2\
#     --weight_ood=50\
#     --weight_targetclass=20\
#     --trigger_size=3\
#     --range=0.1\
#     --epochs=301\
#     --evaluate_mode=overall\
#     --trojan_epochs=301



## Flicker ##
python -u run_adaptive.py \
            --prune_thr=0.4\
            --dataset=Flickr\
            --hidden=128\
            --homo_loss_weight=160\
            --vs_number=160\
            --test_model=GCN\
            --homo_boost_thrd=0.5\
            --train_lr=0.002\
            --lr=0.002\
            --weight_targetclass=10\
            --weight_ood=50\
            --trigger_size=3\
            --range=0.01\
            --epochs=501\
            --rec_epochs=50\
            --evaluate_mode=overall\
            --trojan_epochs=201


