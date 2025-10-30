# 首先是实验部分
python -m main     +name=RE10k     dataset=realestate10k     algorithm=dfot_video_pose     experiment=video_generation     @diffusion/continuous     +trainer.accelerator=npu     +trainer.devices=8     +trainer.num_nodes=1 experiment.training.batch_size=4 experiment.validation.batch_size=6 experiment.test.batch_size=6

# 然后上传训练好的ckpt
python down.py