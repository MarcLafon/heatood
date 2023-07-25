### GMM ##

CUDA_VISIBLE_DEVICES=1 python heat/train_ebm.py \
dataset=cifar100 \
transform=cifar100 \
ebm=heat_gmm_c100 \
backbone=resnet34_c100 \
batch_size=1024 \
n_epochs=20 \
exp_name=heat_gmm_c100 \
ebm.model.steps=20 \
ebm.model.eps_start=1e-3 \
ebm.model.step_size_start=2e-4 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=4 \
eval_freq=2 \
save_freq=1 \
eval_prior=True \
ebm.model.proposal_type=base_dist \
ebm.loss.l2_coef=20 \
accum_iter=1 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=1e3 \
ebm.model.reduce_width=False \


### GMM-std ##

CUDA_VISIBLE_DEVICES=1 python heat/train_ebm.py \
dataset=cifar100 \
transform=cifar100 \
ebm=heat_gmm_std_c100 \
backbone=resnet34_c100 \
batch_size=1024 \
n_epochs=20 \
exp_name=heat_gmm_std_c100 \
ebm.model.steps=20 \
ebm.model.eps_start=1e-3 \
ebm.model.step_size_start=2e-4 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=4 \
eval_freq=2 \
save_freq=1 \
eval_prior=True \
ebm.model.proposal_type=base_dist \
ebm.loss.l2_coef=20 \
accum_iter=1 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=1e3 \
ebm.model.reduce_width=False \
pooling=std

### EL ##

CUDA_VISIBLE_DEVICES=1 python heat/train_ebm.py \
dataset=cifar100 \
transform=cifar100 \
ebm=heat_el_c100 \
backbone=resnet34_c100 \
batch_size=1024 \
n_epochs=8 \
exp_name=heat_el_c100 \
ebm.model.steps=200 \
ebm.model.eps_start=1e-2 \
ebm.model.step_size_start=1e-1 \
ebm.model.use_pcd=False \
eval_freq=2 \
save_freq=1 \
eval_prior=True \
ebm.model.proposal_type=random_normal \
ebm.loss.l2_coef=1 \
accum_iter=1 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=1e0 \
ebm.model.reduce_width=False

