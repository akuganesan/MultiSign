# MultiSign
Multi-Lingual Sign-Language Generation

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
