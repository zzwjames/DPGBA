# DPGBA

**ENVIRONMENTS**

The packages can be installed by directly run the commands in install.sh by 

    bash install.sh

**RUN**

    bash script/train_DPGBA.sh
**NOTES**

1. When there are no defense methods, increasing the value of **weight_target** and **weight_targetclass** will enhance the attack performance.
2. When an outlier detection method is adopted, please also tune the parameter **weight_ood** to achieve a balance between stealthiness and attack performance.

The code is built on [UGBA](https://github.com/ventr1c/UGBA).