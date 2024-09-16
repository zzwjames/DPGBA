
# DPGBA: Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective (KDD 2024)

  

**ENVIRONMENTS**

  

The packages can be installed by directly run the commands in install.sh by

  


    bash install.sh

  

**RUN**

  

    bash script/train_DPGBA.sh

**NOTES**

  

1. Set **'defense_mode=reconstruct'** to introduce outlier detection; set **'defense_mode=none'** for the case of no defense method.

2. When there are no defense methods, increasing the value of **'weight_target'** for $\mathcal{L}_T$ and **'weight_targetclass'** for $\mathcal{L}_E$ will enhance the attack performance.

3. When an outlier detection method is adopted, please also tune the parameter **'weight_ood'** for $\mathcal{L}_D$ to achieve a balance between stealthiness and attack performance.

If you find this repo to be useful, please consider cite our [paper](https://arxiv.org/abs/2405.10757). Thank you.

    @inproceedings{zhang2024rethinking,
      title={Rethinking graph backdoor attacks: A distribution-preserving perspective},
      author={Zhang, Zhiwei and Lin, Minhua and Dai, Enyan and Wang, Suhang},
      booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
      pages={4386--4397},
      year={2024}
    }

  

The code is built on [UGBA](https://github.com/ventr1c/UGBA).
