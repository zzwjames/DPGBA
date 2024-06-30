
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
    --weight_targetclass=3\
    --weight_ood=1\
    --epochs=200\
    --trigger_size=3\
    --trojan_epochs=301


# PubMed ##
python -u run_adaptive.py \
    --dataset=Pubmed\
    --hidden=64\
    --vs_number=40\
    --test_model=GCN\
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
            --dataset=Flickr\
            --hidden=512\
            --vs_number=160\
            --test_model=GCN\
            --train_lr=0.002\
            --lr=0.002\
            --weight_targetclass=10\
            --weight_ood=50\
            --trigger_size=3\
            --range=0.01\
            --epochs=1501\
            --trojan_epochs=201







