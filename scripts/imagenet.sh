### GMM ##

python heat/train_ebm.py \
dataset=imagenet_features \
transform=imagenet \
ebm=heat_gmm_imagenet \
backbone=resnet50_imagenet \
batch_size=256 \
n_epochs=8 \
exp_name=heat_gmm_imagenet \
ebm.model.steps=10 \
ebm.model.eps_start=1e-4 \
ebm.model.step_size_start=1e-7 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=6 \
ebm.model.sample_from_batch_statistics=True \
ebm.model.base_dist.use_simplified_mahalanobis_score=True \
eval_freq=2 \
save_freq=1 \
eval_prior=False \
ebm.model.proposal_type=base_dist_temp \
ebm.loss.l2_coef=10 \
accum_iter=20 \
ebm.optimizer.opt.lr=1e-4 \
ebm.optimizer.sch.milestones=[1] \
ebm.optimizer.sch.gamma=0.1 \
ebm.model.temperature_prior=5e3 \
ebm.model.reduce_width=True


### GMM-std ##

python heat/train_ebm.py \
dataset=imagenet_features \
transform=imagenet \
ebm=heat_gmm_std_imagenet \
backbone=resnet50_imagenet \
batch_size=256 \
n_epochs=8 \
exp_name=heat_gmm_std_imagenet \
ebm.model.steps=0 \
ebm.model.eps_start=1e-4 \
ebm.model.step_size_start=1e-4 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=6 \
ebm.model.sample_from_batch_statistics=True \
ebm.model.base_dist.use_simplified_mahalanobis_score=True \
eval_freq=2 \
save_freq=1 \
eval_prior=False \
ebm.model.proposal_type=base_dist_temp \
ebm.loss.l2_coef=40 \
accum_iter=20 \
ebm.optimizer.opt.lr=1e-4 \
ebm.optimizer.sch.milestones=[1] \
ebm.optimizer.sch.gamma=0.1 \
ebm.model.temperature_prior=5e3 \
ebm.model.reduce_width=True \
pooling=std

### EL ##

python heat/train_ebm.py \
dataset=imagenet_features \
transform=imagenet \
ebm=heat_el_imagenet \
backbone=resnet50_imagenet \
batch_size=256 \
n_epochs=8 \
exp_name=heat_el_imagenet \
ebm.model.steps=40 \
ebm.model.eps_start=1e-3 \
ebm.model.step_size_start=1e3 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=4 \
eval_freq=2 \
save_freq=1 \
eval_prior=True \
ebm.model.proposal_type=random_normal \
ebm.loss.l2_coef=20 \
accum_iter=10 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=5e2 \
ebm.model.reduce_width=True
