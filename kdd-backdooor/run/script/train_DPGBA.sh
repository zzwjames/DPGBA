
models=(GCN)




# Cora ##
python -u run_adaptive.py \
    --target_class=2\
    --dataset=Cora\
    --k=50\
    --vs_number=10\
    --hidden=128\
    --train_lr=0.005\
    --lr=0.005\
    --test_model=GCN\
    --weight_target=1\
    --weight_targetclass=2\
    --weight_ood=1\
    --epochs=200\
    --trigger_size=3\
    --trojan_epochs=301


# PubMed ##
python -u run_adaptive.py \
    --prune_thr=0.2\
    --dataset=Pubmed\
    --hidden=64\
    --homo_loss_weight=160\
    --vs_number=40\
    --test_model=GCN\
    --homo_boost_thrd=0.5\
    --train_lr=0.002\
    --lr=0.002\
    --target_class=2\
    --weight_ood=2\
    --weight_targetclass=5\
    --trigger_size=3\
    --range=0.1\
    --epochs=301\
    --trojan_epochs=301



# # ## Flicker ##
python -u run_adaptive.py \
            --prune_thr=0.4\
            --dataset=Flickr\
            --hidden=512\
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
            --epochs=1501\
            --trojan_epochs=201




