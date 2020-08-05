# Forced Variational Integrator Networks

Requires Python3, gym, torch.

Install custom gym environments
```
cd gym_custom
pip3 install -e .
```

Train a VV-FVIN model on the Damped Pendulum and save model to
"models/vv_pendulum.pt"
```
python3 VI_train.py --save_name vv_pendulum --epochs 10000 --n_traj 5 --model_type VI_VV_model --env PendulumMod-v0
```
For the ResNN baseline run
```
python3 VI_train.py --save_name res_pendulum --epochs 10000 --n_traj 5 --model_type Res_model --env PendulumMod-v0
```

Plot either or both results with no control
```
python3 VI_evaluate.py --H 100 --model1  res_pendulum --model2 vv_pendulum
```

To run additional n iterations with CEM control simply add the option
--iterations n. To run with mpc control add --policy mpc. To run random
control add --policy random

