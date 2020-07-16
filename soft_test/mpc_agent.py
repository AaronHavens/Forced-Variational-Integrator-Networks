from cem import CEM
from data         import Pose, Pressure, TransmitData
from gripcheck    import GripCheck
from numpy.linalg import norm
import numpy as np
import torch

class Agent():
    def __init__(
            self,
            base_path='../models',
            step_limit=40,
            max_err=1.0,
            models=(9, 15, 20)):

        self.base_path = base_path
        self.step_limit = step_limit
        self.max_err = max_err
        self.models = np.array(models)
        self.step_count = 0

        # Set up RL model
        action_high = np.array([1,1,1])#,1])
        action_low  = np.array([-1,-1,-1])#,0])
        state_high = np.array([np.inf, np.inf, np.inf, 40, 44])#, 20e-2])
        state_low = np.array([-np.inf, -np.inf, -np.inf, 11, -44])#, 0])
        model = torch.load('models/res_test.pt')
        upper_bound = np.array([
        self._rl = CEM(model,K=500,T=10,state_high, state_low, action_high, action_low)
        #self._rl = Model(inputs=S, outputs=V)

        self._grip_check = GripCheck(f'{self.base_path}/gripper_data.txt')

        self._temp_count = 0
        self._last_ext = None
        self._last_pres = Pressure()

    def reach(self, pose, pres, ext):
        print('Error:', pose)
        print('Input:', pres)
        print('Ext:', ext)
        print('Iteration count:', self.step_count)

        tx_data = self._status(pose)

        if not tx_data.done:
            self._load_weights(ext)
            state = np.hstack((pose.x, pose.y, pose.z, pres.bend, pres.rot))

            if self._temp_count == 0:
                #initial pressures read at (0, 0); Should be fixed in LV
                state[3]= 12.0
                state[4]= 0.0
                self._temp_count += 1

            pres_delta = self._rl.predict(state)
            #pres_delta = self._rl.predict(state.reshape(1, state.shape[0]))
            scaled_pres_delta = self._scale_actions(pres_delta)

            # Unconstrained command pressures
            tx_data.pres.bend = round(pres.bend + scaled_pres_delta[0, 0], 1)
            tx_data.pres.rot = round(pres.rot + scaled_pres_delta[0, 1], 1)

            tx_data.pres.bend = min(40, max(11, tx_data.pres.bend))
            tx_data.pres.rot = min(44, max(-44, tx_data.pres.rot))

            self.step_count += 1
            self._last_pres = tx_data.pres

        return tx_data
    
    def _status(self, pose):
        err = norm(pose.to_list())
        print(f'Norm: {err:.4f}')
        reached = err < self.max_err
        done = reached or self.step_count == self.step_limit
        if done:
            acquired = self._grip_check.check()
        else:
            acquired = False

        if done:
            self.step_count = 0

        return TransmitData(self._last_pres, done, reached, acquired)

    def _scale_actions(self, output):
        scaled_actions = np.zeros((1, 2))
        scaled_actions[0][0] = 1.0 * (-11 + 29 * (output[0][0] + 1) / 2.0)
        scaled_actions[0][1] = 1.0 * (-44 + 88 * (output[0][1] + 1) / 2.0)

        return scaled_actions

    def _load_weights(self, ext):
        if ext == self._last_ext:
            return

        self._last_ext = ext

        model_ext = self.models[np.argmin(np.abs(self.models - ext))]
        print(f'Loading RL model weights for extrusion length {model_ext}...')
        self._rl.load_weights(f'{self.base_path}/actormodel{model_ext}_h.h5')

