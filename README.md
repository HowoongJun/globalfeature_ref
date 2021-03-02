# Global Feature References
Reference codes for global feature


## NetVLAD

* [NetVLAD: CNN architecture for weakly supervised place recognition, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf)

This reference code is based on the code of [uzh-rpg](https://github.com/uzh-rpg/netvlad_tf_open), which was [originally implemented in Matlab](https://github.com/Relja/netvlad).
Tensorflow checkpoint is based on the [uzh-rpg](https://github.com/uzh-rpg/netvlad_tf_open), so you have to [download it through their github repository](http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip) and locate the file in the `/globalfeature_ref/netvlad/checkpoints/` folder. 

We modified their work for our own [global localization framework](https://github.com/HowoongJun/visualloc_global). 

* * *

## GeM (Generalized-Mean)

* [Fine-tuning CNN Image Retrieval with No Human Annotation](https://cmp.felk.cvut.cz/~radenfil/publications/Radenovic-TPAMI18.pdf)

This reference code is based on the [code of Filip Randenović et al.](https://github.com/filipradenovic/cnnimageretrieval-pytorch)