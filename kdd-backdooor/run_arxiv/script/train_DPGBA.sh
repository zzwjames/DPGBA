
python -u run_adaptive.py \
                    --prune_thr=0.8\
                    --dataset=ogbn-arxiv\
                    --homo_loss_weight=500\
                    --vs_number=565\
                    --hidden=32\
                    --test_model=GCN\
                    --selection_method=none\
                    --homo_boost_thrd=0.95\
                    --weight_targetclass=1\
                    --weight_target=25\
                    --epochs=500\
                    --k=10\
                    --trigger_size=3\
                    --weight_ood=50\
                    --rec_epochs=300\
                    --range=0.1\
                    --trojan_epochs=81