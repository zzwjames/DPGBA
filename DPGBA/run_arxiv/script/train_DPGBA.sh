
python -u run_adaptive.py \
                    --dataset=ogbn-arxiv\
                    --homo_loss_weight=500\
                    --vs_number=565\
                    --hidden=32\
                    --test_model=GCN\
                    --selection_method=none\
                    --weight_targetclass=1\
                    --weight_target=25\
                    --epochs=500\
                    --k=10\
                    --trigger_size=3\
                    --weight_ood=50\
                    --rec_epochs=300\
                    --range=0.1\
                    --trojan_epochs=81