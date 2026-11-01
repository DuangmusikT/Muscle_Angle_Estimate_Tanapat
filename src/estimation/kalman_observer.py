"""Multi-rate 1D Kalman filter for joint angle (p) and speed (s)."""

import numpy as np


class KinematicKalmanFilter:
    """Constant-velocity Kalman filter with state x = [p, s].

    Predict step runs at the feature rate (e.g. 30 Hz). Correction steps
    can be applied from sEMG multi-task outputs (p, s) and/or from
    MediaPipe vision angle (p only), possibly at a different rate.
    """

    def __init__(self, dt=1.0/30.0, q_p=1e-2, q_s=1e-1, r_p=5.0, r_s=10.0, r_vision=2.0):
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = np.array([[q_p, 0.0], [0.0, q_s]])
        self.H_semg = np.eye(2)
        self.H_vis = np.array([[1.0, 0.0]])
        self.R_semg = np.array([[r_p, 0.0], [0.0, r_s]])
        self.R_vis = np.array([[r_vision]])

        self.x = np.zeros(2)   # [p, s]
        self.P = np.eye(2) * 10.0
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_semg(self, p_hat, s_hat):
        z = np.array([p_hat, s_hat])
        y = z - self.H_semg @ self.x
        S = self.H_semg @ self.P @ self.H_semg.T + self.R_semg
        K = self.P @ self.H_semg.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H_semg) @ self.P
        self.initialized = True

    def update_vision(self, p_vis):
        z = np.array([p_vis])
        y = z - self.H_vis @ self.x
        S = self.H_vis @ self.P @ self.H_vis.T + self.R_vis
        K = self.P @ self.H_vis.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H_vis) @ self.P
        self.initialized = True

    @property
    def p(self):
        return float(self.x[0])

    @property
    def s(self):
        return float(self.x[1])

    def reset(self, p0=0.0, s0=0.0):
        self.x = np.array([p0, s0], dtype=float)
        self.P = np.eye(2) * 10.0
        self.initialized = False
