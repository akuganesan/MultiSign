# MultiSign
Multi-Lingual Sign-Language Generation

### TODO
- Right now because we only have one subject I manually split the current subject into a training and validation dataset, this will have to be "unhard-coded" 
- Add visualization code to the training script
- Add abitlity to load model/checkpoint from a previous run and resume training
- Write/fix code for evaluation and testing (inside runner.py)

### training script (src/train.py) - April 4, 2021
The default way to run this script right now is the following line (this uses the default hyperparameters for training and names the run "test"
```bash
$ python train.py --config config/basic.txt
```
When the script runs it will also save a copy of the config and args used for that experiment. For training, it performs evalutation after each epoch and saves the model and optimizer for the model with the lowest validation and the same things for the latest epoch.

### Skeleton Format
index 0 to 14 (inclusive) -> BODY
index 15 to 56 (inclusive) -> HANDS

#### Using dataset.py
For additional information please checkout the [dataloader_example.ipynb](https://github.com/akuganesan/MultiSign/blob/main/dataloader_example.ipynb)
```python
test_dataset = SIGNUMDataset('/scratch/datasets/SIGNUM', use_pose=True, subsample=10)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=test_dataset.collate)

for i, data in enumerate(test_dataloader):
    img_seq = data['img_seq']
    pose_seq = data['pose_seq']
    transl_eng = data['transl_eng']
    transl_deu = data['transl_deu']
    print(img_seq.shape)
    print(pose_seq.shape)
```
