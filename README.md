#point-cloud-cross-domain
Configurations for cross domain evaluation and transfer learning

For use with https://github.com/open-mmlab/OpenPCDet.

Install requirements (python==3.9.17):
```
pip install -r requirements.txt
```

Clone OpenPCDet:
```
git clone https://github.com/open-mmlab/OpenPCDet
```
Follow repository instrucitons to install package.

Download data for KITTI from:
https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev

Download data for ONCE from:
https://once-for-auto-driving.github.io/

Follow instructions in OpenPCDet to organise KITTI data.
For custom setup of ONCE dataset, move the downloaded ONCE data to ./data
And run:
```
python create_custom_dataset_from_once.py
```
After this command all the data will be created in: OpenPCDet/data/custom_once/
Overwrite the OnceCustomDataset class definitions to pcdet:
```
cp custom_dataset OpenPCDet/pcdet/datasets/custom
```

From this point use the instructions in OpenPCDet to run the trianing and evaluation.

For only ONCE models use the cfgs/once/ configurations.

For cross domain evaluation use the cfgs/cross configurations. They are similiar but using separate configurations makes separting results and checkpoints easier.

Thanks to:
```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```