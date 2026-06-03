import numpy as np
from utils import convert as cvt
from utils.axxb import axxb
from scipy.spatial.transform import Rotation as R
from utils.bpoly import BPoly


class calibration:
    def __init__(self, bp_t_order=2, bp_r_order=3):
        self.t2f = np.array([31.248, 0, 25]) * 1e-3
        self.r2t = np.array([608.5, 0, 13]) * 1e-3
        self.d2r = np.array([0, 0, 76.14070118]) * 1e-3
        self.bp_t = BPoly(3, 6, order=bp_t_order)
        self.bp_r = BPoly(2, 6, order=bp_r_order)
        self.iter = 0

    def to_delta_pos(self, fk, jp):
        n_data = fk.shape[0]
        delta_pos = np.zeros((n_data, 3))
        for i in range(n_data):
            rotX = R.from_euler("XYZ", [jp[i, 3], 0, 0])
            rotY = R.from_euler("XYZ", [0, jp[i, 4], 0])
            rot_pos = self.d2r + rotX.apply(self.r2t + rotY.apply(self.t2f))
            delta_pos[i, :] = fk[i, :3] - rot_pos
        return delta_pos

    def fit_calibrate_t(self, fk, jp, ee):
        T_fk_encoder = cvt.vect7_htm(fk)
        T_ee = cvt.vect7_htm(ee)
        T_fk_optical = np.zeros(T_fk_encoder.shape)
        n_data = fk.shape[0]
        for i in range(n_data):
            T_fk_optical[i, :, :] = (
                cvt.inv(self.T_b) @ T_ee[i, :, :] @ cvt.inv(self.T_tip)
            )
        fk_delta = self.to_delta_pos(fk, jp)
        # for translation only calibration, use pose difference as output
        T_fk_diff = np.zeros(T_fk_encoder.shape)
        for i in range(n_data):
            T_fk_diff[i, :, :] = cvt.inv(T_fk_encoder[i, :, :]) @ T_fk_optical[i, :, :]
        fk_diff = cvt.htm_vect6(T_fk_diff)
        self.bp_t.fit(fk_delta, fk_diff, 0.1)

    def fit_calibrate_r(self, fk, jp, ee):
        T_fk_encoder = cvt.vect7_htm(fk)
        T_ee = cvt.vect7_htm(ee)
        T_fk_optical = np.zeros(T_fk_encoder.shape)
        n_data = fk.shape[0]
        for i in range(n_data):
            T_fk_optical[i, :, :] = (
                cvt.inv(self.T_b) @ T_ee[i, :, :] @ cvt.inv(self.T_tip)
            )
        # for rotation only calibration, use only jp3 and jp4 as input
        jp_wrist = jp[:, 3:]
        # for rotation only calibration, use pose difference as output
        T_fk_diff = np.zeros(T_fk_encoder.shape)
        for i in range(n_data):
            T_fk_diff[i, :, :] = cvt.inv(T_fk_encoder[i, :, :]) @ T_fk_optical[i, :, :]
        fk_diff = cvt.htm_vect6(T_fk_diff)
        self.bp_r.fit(jp_wrist, fk_diff, 0.1)

    def calibrate_t(self, fk, jp):
        n_data = fk.shape[0]
        fk_delta = self.to_delta_pos(fk, jp)
        fk_calibrate = np.zeros((n_data, 7))
        for i in range(n_data):
            T_fk = cvt.vect7_htm(fk[i, :])
            T_fk_cal = T_fk @ cvt.vect6_htm(self.bp_t(fk_delta[i, :].reshape((1, 3))))
            fk_calibrate[i, :] = cvt.htm_vect7(T_fk_cal)
        return fk_calibrate

    def calibrate_r(self, fk, jp):
        n_data = fk.shape[0]
        fk_calibrate = np.zeros((n_data, 7))
        for i in range(n_data):
            T_fk = cvt.vect7_htm(fk[i, :])
            T_fk_cal = T_fk @ cvt.vect6_htm(self.bp_r(jp[i, 3:5].reshape((1, 2))))
            fk_calibrate[i, :] = cvt.htm_vect7(T_fk_cal)
        return fk_calibrate

    def calibrate(self, fk, jp):
        return self.calibrate_r(self.calibrate_t(fk, jp), jp)

    def head_eye(self, fk, ee):
        T_ee = cvt.vect7_htm(ee)
        T_fk = cvt.vect7_htm(fk)
        A = np.zeros((T_ee.shape[0] - 1, 4, 4))
        B = np.zeros(A.shape)
        n_data = fk.shape[0]
        for i in range(n_data - 1):
            A[i, :, :] = cvt.inv(T_fk[0, :, :]) @ T_fk[i + 1, :, :]
            B[i, :, :] = cvt.inv(T_ee[0, :, :]) @ T_ee[i + 1, :, :]
        a = cvt.htm_vect7(A)
        b = cvt.htm_vect7(B)
        self.T_tip = axxb(a, b)
        self.T_b = T_ee[0, :, :] @ cvt.inv(self.T_tip) @ cvt.inv(T_fk[0, :, :])

        # Print RMSE for debugging
        T_fk_calibrated = np.zeros(T_fk.shape)
        for i in range(n_data):
            T_fk_calibrated[i, :, :] = (
                cvt.inv(self.T_b) @ T_ee[i, :, :] @ cvt.inv(self.T_tip)
                )
        fk_calibrated = cvt.htm_vect6(T_fk_calibrated)
        fk_encoder = cvt.htm_vect6(T_fk)

        # Only for translation error
        rmse = np.sqrt(np.mean((fk_calibrated[:, :3] - fk_encoder[:, :3]) ** 2, axis=0))
        print("RMSE:", rmse)
