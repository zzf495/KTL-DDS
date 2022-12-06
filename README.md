# Knowledge Transfer Learning via Dual Density Sampling for Resource-Limited Domain Adaptation (KTL-DDS)

This repository contains the codes of the  KTL-DDS.

## Code files (matlab implementation)

├─+func: The functions used for `DDSA`. For the other algorithms, please refer to our [repository](https://github.com/zzf495/Re-implementations-of-SDA).  
├─data  
│  ├─Adaptiope20: The `.mat` files of Adaptiope20 (ResNet50) gained by using a ResNet50 extractor proposed in [1].    
│  │						    Only the first 20 classes are used in the paper.  
│  └─Office31: The `.mat` files of Office31 (ResNet50) gained by using a ResNet50 extractor proposed in [1].  
└─util: The folder that stores `KTL-DDS`, `DDSA` and some auxiliary functions.  
      └─liblinear-2.30: A libsvm tool.  

  

> [1] Wang, Qian, and Toby Breckon. "Unsupervised domain adaptation via structured prediction based selective pseudo-labeling." *Proc. AAAI Conf. Artif. Intell.*. Vol. 34. No. 04. 2020.  

## Citation

The article has been accepted for publication in a future issue of "IEEE/CAA J. Autom. Sinica".  

> @article{zheng2022domain,  
>   	title={Knowledge Transfer Learning via Dual Density Sampling for Resource-Limited Domain Adaptation},  
>   	author={Zheng, Zefeng and Teng, Luyao and Zhang, Wei and Wu, Naiqi and Teng, Shaohua},  
>   	journal={IEEE/CAA J. Autom. Sinica},  
>   	volume={},  
>   	number={},  
>   	pages={},  
>   	year={},  
>   	note={In press},  
> }