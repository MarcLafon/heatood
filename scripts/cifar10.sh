### GMM ##

python heat/train_ebm.py \
dataset=cifar10 \
transform=cifar10 \
ebm=heat_gmm_c10 \
backbone=resnet34_c10 \
batch_size=1024 \
n_epochs=20 \
exp_name=heat_gmm_c10 \
ebm.model.steps=40 \
ebm.model.eps_start=5e-3 \
ebm.model.step_size_start=1e-4 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=4 \
eval_freq=2 \
save_freq=1 \
eval_prior=True \
ebm.model.proposal_type=base_dist \
ebm.loss.l2_coef=10 \
accum_iter=1 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=1e3 \
ebm.model.reduce_width=False


### GMM-std ##

python heat/train_ebm.py \
dataset=cifar10 \
transform=cifar10 \
ebm=heat_gmm_std_c10 \
backbone=resnet34_c10 \
batch_size=1024 \
n_epochs=20 \
exp_name=heat_gmm_std_c10 \
ebm.model.steps=1 \
ebm.model.eps_start=5e-3 \
ebm.model.step_size_start=1e-4 \
ebm.model.use_pcd=False \
ebm.model.hidden_dim=1024 \
ebm.model.n_hidden_layers=4 \
ebm.model.proposal_type=base_dist \
ebm.loss.l2_coef=10 \
accum_iter=1 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=1e3 \
ebm.model.reduce_width=False \
pooling=std

### EL ##

python heat/train_ebm.py \
dataset=cifar10 \
transform=cifar10 \
ebm=heat_el_c10 \
backbone=resnet34_c10 \
batch_size=256 \
n_epochs=20 \
exp_name= heat_el_c10 \
ebm.model.steps=50 \
ebm.model.eps_start=1e-1 \
ebm.model.step_size_start=1e-1 \
ebm.model.use_pcd=False \
eval_freq=2 \
save_freq=1 \
eval_prior=True \
ebm.model.proposal_type=random_normal \
ebm.loss.l2_coef=1e-1 \
accum_iter=1 \
ebm.optimizer.opt.lr=1e-5 \
ebm.model.temperature_prior=1e0 \
ebm.model.reduce_width=False
