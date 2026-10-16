"""Single-trait additive GBLUP, intercept + training-only REML.

Genotypes use MENET's {-1, 0, 1} allele coding (fractional dosage allowed).
Relationship scaling follows rrBLUP A.mat's unshrunk additive matrix.
This is not pedigree BLUP, ssGBLUP, or a multi-environment mixed model.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


class GBLUPRegressor:
    def __init__(self, max_memory_bytes=2 * 1024**3):
        self.max_memory_bytes = int(max_memory_bytes)

    @staticmethod
    def _check_x(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError("GBLUP 要求非空、有限的二维基因型矩阵")
        if x.min() < -1.000001 or x.max() > 1.000001:
            raise ValueError("GBLUP 当前要求 [-1,1] 剂量编码；不自动猜测 0/1/2 编码")
        return x

    def fit(self, x, y):
        x = self._check_x(x)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, m = x.shape
        if n < 3 or len(y) != n or not np.isfinite(y).all():
            raise ValueError("GBLUP 至少需要 3 个有效训练样本及匹配的表型")
        # Conservative estimate; no silent sample/marker subsampling.
        estimated = 8 * (4 * n * n + 2 * n * m)
        if estimated > self.max_memory_bytes:
            raise MemoryError(f"GBLUP 估算工作内存 {estimated} 字节超过限制 {self.max_memory_bytes}")
        self.n_features_in_ = m
        self.marker_mean_ = x.mean(axis=0)
        self.active_markers_ = x.var(axis=0) > 1e-12
        if not self.active_markers_.any():
            raise ValueError("GBLUP 训练集没有多态标记")
        p = (self.marker_mean_[self.active_markers_] + 1.0) / 2.0
        self.relationship_scale_ = float(2.0 * np.sum(p * (1.0 - p)))
        if self.relationship_scale_ <= 0:
            raise ValueError("GBLUP 加性关系矩阵缩放常数无效")
        z = x[:, self.active_markers_] - self.marker_mean_[self.active_markers_]
        kernel = (z @ z.T) / self.relationship_scale_
        eigenvalues, eigenvectors = eigh(kernel, check_finite=False)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        projected = eigenvectors.T @ yc
        df = n - 1

        def objective(log_delta):
            delta = np.exp(log_delta)
            residual = float(np.sum(projected**2 / (eigenvalues + delta)))
            # K is centered, so the intercept removes one zero-eigenvalue mode.
            logdet = np.log(eigenvalues + delta).sum() - log_delta
            return float(df * np.log(max(residual / df, np.finfo(float).tiny)) + logdet)

        if np.dot(yc, yc) < 1e-20:
            log_delta = 20.0
            self.fit_status_ = "constant_phenotype"
        else:
            result = minimize_scalar(objective, bounds=(-20.0, 20.0), method="bounded")
            if not result.success or not np.isfinite(result.fun):
                raise ValueError("GBLUP REML 方差参数估计未收敛")
            choices = [float(result.x), -20.0, 20.0]
            log_delta = min(choices, key=objective)
            self.fit_status_ = "boundary" if abs(log_delta) >= 19.99 else "converged"
        self.delta_ = float(np.exp(log_delta))
        alpha = eigenvectors @ (projected / (eigenvalues + self.delta_))
        self.marker_effects_ = z.T @ alpha / self.relationship_scale_
        self.genetic_variance_ = float(max(np.dot(yc, alpha) / df, 0.0))
        self.residual_variance_ = self.genetic_variance_ * self.delta_
        return self

    def predict(self, x):
        x = self._check_x(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("GBLUP 预测标记数量与训练不一致")
        z = x[:, self.active_markers_] - self.marker_mean_[self.active_markers_]
        return self.intercept_ + z @ self.marker_effects_
