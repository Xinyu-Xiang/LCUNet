# LCUNet
## 1st Solution of MMDD 2025: The 1st Multimodal Deception Detection Competition @MM2025
LCUNet: A Lightweight Concatenated Unified Mapping Multi-modal Deception Detector  (*^▽^*)

This is official Pytorch implementation of "[LCUNet: A Lightweight Concatenated Unified Mapping Multi-modal Deception Detector](https://dl.acm.org/doi/10.1145/3728425.3759923)"

## Framework
![The overall framework of the proposed LCUNet algorithm.](https://github.com/Xinyu-Xiang/LCUNet/blob/main/assets/framework.pdf)


## Team
```
Team: WHU_PB
Team leader: Glenn_xxy (Real name: Xiang Xinyu)
```

## Before running:
```
Place the training and test datasets inside the ./dataset2/
put our ckpt is in ./checkpoint/
```

## Train:
```bash
sh train_test_feature_phase2_upload.sh
```

## Test:
```bash
python test_phase2_final_upload.py
```
## Environment
Please check the ``environment.yml `` for details

## Checkpoint
https://pan.baidu.com/s/1Xr5-DGyH48NhR0xg64R2zQ?pwd=uiaf   
pw: uiaf

## Ours result:
Please check the ./result

## More details:
If there are any problems with the operation, please feel free to contact me at any time. 
I will be happy to answer any questions you have.
```bash
Email me:
xiangxinyu@whu.edu.cn 
1977587176@qq.com
```

## Thanks:
```bash
@inproceedings{guo2023audio,
  title={Audio-visual deception detection: Dolos dataset and parameter-efficient crossmodal learning},
  author={Guo, Xiaobao and Selvaraj, Nithish Muthuchamy and Yu, Zitong and Kong, Adams Wai-Kin and Shen, Bingquan and Kot, Alex},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22135--22145},
  year={2023}
}
@article{guo2024benchmarking,
  title={Benchmarking Cross-Domain Audio-Visual Deception Detection},
  author={Guo, Xiaobao and Yu, Zitong and Selvaraj, Nithish Muthuchamy and Shen, Bingquan and Kong, Adams Wai-Kin and Kot, Alex C},
  journal={arXiv preprint arXiv:2405.06995},
  year={2024}
}
```

Their work has significantly contributed to our research and implementation.

## If this work is helpful to you, please cite it as：
```
@inproceedings{xiang2025lcunet,
  title={LCUNet: A Lightweight Concatenated Unified Mapping Multi-modal Deception Detector},
  author={Xiang, Xinyu and Li, Shengxiang and Huang, Jun and Yan, Qinglong and Zhu, Zhenjie and Zhang, Hao and Ma, Jiayi},
  booktitle={Proceedings of the 1st International Workshop \& Challenge on Subtle Visual Computing},
  pages={46--51},
  year={2025}
}
```




