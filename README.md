# fICHnet: Predict long term outcomes after ICH
* installation
```bash
git clone https://github.com/Yutong441/fICHnet
# recommend to install to a virtual environment
pip install -e .
```

Then download the model weights from [here](https://drive.google.com/drive/folders/1i9GzJ47m1qRmAvRjSgJuYUFXpbhK2XR4?usp=sharing).
(NB: I will need to approve your access after you clicked the link).
It can be stored at any location you chose, as long as you know the path.

Download some additional data from [here](https://drive.google.com/drive/folders/1xUZ8MuiDCIMoiA4WhgTor7me-mvPSQL7?usp=sharing).
Save the folder in the `data/` directory.

* usage:
    - need to put the absolute paths of all the nifti images to a single master
    file and supply the path of that master file to the `--img_paths` flag.
    - images should be original non-contrast CT images without any preprocessing
    - the path to the directory containing the model weight goes to the
    `--model_path` flag.
    - script to run on HPC: [GPU version](./test/sample_gpu.sh), [CPU version](./test/sample_cpu.sh)
    - computation time per patient: 2.4s for preprocessing (4 CPU), 0.01s for
    prediction on GPU, 0.05s for prediction on CPU

```bash
python predict.py --img_paths=results/img_path.txt \
    --model_path=models/ \
    --save_dir=results/ \
    --device="cuda:0" # for gpu, "cpu" for cpu
```
    
* output
    - 3 files in the folder specified by the `--save_dir` flag:
    `pred_dependent.csv`, `pred_disability.csv`, `pred_severe_disability.csv`
    - each contains the probability of a morbidity-free survival at various time
    points after ICH
