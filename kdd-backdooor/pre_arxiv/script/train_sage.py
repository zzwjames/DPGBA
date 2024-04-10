models=(GCN)
# defense_modes=(reconstruct)
seeds=(12 15 16 17 18 19 20 21 22 23 24 25)

target_classes=(2)

# for defense_mode in ${defense_modes[@]};
# do 
    for model in ${models[@]};
    do
        for target_class in ${target_classes[@]};
        do
            for seed in ${seeds[@]};
            do
                # python -u run_adaptive.py \
                #     --prune_thr=0.8\
                #     --dataset=ogbn-arxiv\
                #     --homo_loss_weight=500\
                #     --vs_number=565\
                #     --hidden=64\
                #     --test_model=${model}\
                #     --selection_method=none\
                #     --homo_boost_thrd=0.95\
                #     --weight_targetclass=10\
                #     --weight_target=1\
                #     --epochs=500\
                #     --k=20\
                #     --seed=${seed}\
                #     --trigger_size=3\
                #     --weight_ood=50\
                #     --rec_epochs=250\
                #     --range=0.1\
                #     --target_class=2\
                #     --trojan_epochs=101
                python -u run_adaptive.py \
                    --prune_thr=0.8\
                    --dataset=ogbn-arxiv\
                    --homo_loss_weight=500\
                    --vs_number=565\
                    --hidden=64\
                    --test_model=${model}\
                    --selection_method=none\
                    --homo_boost_thrd=0.95\
                    --weight_targetclass=10\
                    --weight_target=1\
                    --epochs=500\
                    --k=20\
                    --seed=${seed}\
                    --trigger_size=3\
                    --weight_ood=50\
                    --rec_epochs=250\
                    --range=0.1\
                    --target_class=2\
                    --trojan_epochs=81
            done
        done
    done    
# done