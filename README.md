# A Mathematical Analysis of Learning Loss for Active Learning in Regression

[CVF Open Access - LearningLoss++](https://openaccess.thecvf.com/content/CVPR2021W/TCV/html/Shukla_A_Mathematical_Analysis_of_Learning_Loss_for_Active_Learning_in_CVPRW_2021_paper.html) <br>
Presented at the IEEE CVPR Workshop on Fair, _Data Efficient_ and Trusted Computer Vision. <br>
This repository contains the code for various active learning methods used in Human Pose Estimation.

The code can be executed after setting up the configuration file  `configuration.yml` and running `python main.py`

If this repository is helpful, please do consider starring, and citing:
```
@InProceedings{Shukla_2021_CVPR,
    author    = {Shukla, Megh and Ahmed, Shuaib},
    title     = {A Mathematical Analysis of Learning Loss for Active Learning in Regression},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {3320-3328}
}
```

< ReadMe in progress, will update with examples of `configuration.yml` and running the code.  > <br>
Please feel free to contact megh.shukla@daimler.com for any queries and suggestions! 

If you liked this paper, you may also like [EGL++ (arXiv)](https://arxiv.org/abs/2104.09493), our latest work on Active Learning for Human Pose Estimation. <br>
The paper shows that the classical Expected Gradient Length algorithm is equivalent to Bayesian uncertainty under the linear regression framework! <br>
The paper then proposes EGL++, a heuristic that is competitive with other SOTA active learning algorithms for Human Pose Estimation.