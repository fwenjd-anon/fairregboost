import numpy as np
from scipy.optimize import minimize

def f_dis(Y, L):
    bins = np.linspace(0, 1, L) - (1 / (2 * (L - 1)))
    bins = np.append(bins, 1)  # Ensure the last bin includes the upper limit
    Yd = (np.digitize(Y, bins) - 1) / (L - 1)
    return Yd


def f_err(YT, YF):
    return np.mean((YT - YF) ** 2)


def f_fai(Y, S):
    tt = np.linspace(np.min(Y), np.max(Y), 1000)
    vv = np.unique(S)
    nn = [np.sum(S == vv[0]), np.sum(S == vv[1])]
    fai = 0
    for t in tt:
        fai = max(fai, abs(
            np.sum(Y[S == vv[0]] <= t) / nn[0] - 
            np.sum(Y[S == vv[1]] <= t) / nn[1]
        ))
    return fai


def f_ks(Y, S):
    tt = np.linspace(np.min(Y), np.max(Y), 1000)
    vv = np.unique(S)
    nn = [np.sum(S == vv[0]), np.sum(S == vv[1])]
    fai = 0
    for t in tt:
        fai = max(fai, abs(
            np.sum(Y[S == vv[0]] <= t) / nn[0] - 
            np.sum(Y[S == vv[1]] <= t) / nn[1]
        ))
    return fai


def f_lambda(Y, S, M, L, beta):
    def f_mincon(lambda_, etaX, S, M, L, beta):
        p = [np.mean(S == 0), np.mean(S == 1)]
        ris = 0
        for s in [0, 1]:
            for ix in range(len(etaX)):
                if S[ix] == s:
                    tot = 0
                    for i in range(-L, L + 1):
                        h = 2 * p[s] * i * M * etaX[ix] / L - p[s] * i**2 * M**2 / L**2
                        tot += np.exp(lambda_[i + L] * (2 * s - 1) / beta + h / beta)
                    for i in range(-L, L + 1):
                        h = 2 * p[s] * i * M * etaX[ix] / L - p[s] * i**2 * M**2 / L**2
                        tmp1 = np.exp(lambda_[i + L] * (2 * s - 1) / beta + h / beta) / tot
                        ris += tmp1 * (
                            (2 * s - 1) * lambda_[i + L] + h - beta * np.log((2 * L + 1) * tmp1)
                        ) / np.sum(S == s)
        return ris
    
    initial_lambda = np.zeros(2 * L + 1)
    bounds = [(0, 4 * M) for _ in range(2 * L + 1)]
    result = minimize(f_mincon, initial_lambda, args=(Y, S, M, L, beta), bounds=bounds)
    return result.x


def f_ICML(YL, YT, SL, ST, L, beta):
    YL = YL * 2 - 1
    YT = YT * 2 - 1
    M = 1
    L = L // 2
    p = [np.mean(SL == 0), np.mean(SL == 1)]
    lambda_ = f_lambda(YL, SL, M, L, beta)
    YTd = np.zeros_like(YT)
    i = np.arange(-L, L + 1)
    for k in range(len(YT)):
        v = p[int(ST[k])] * (YT[k] - i * M / L)**2 + (1 - 2 * ST[k]) * lambda_
        j = np.argmin(v)
        YTd[k] = i[j] * M / L
    return (YTd + 1) / 2


def f_NIPS(YL, YT, SL, ST):
    u = np.unique(SL)
    nM, iM = max([(np.sum(SL == u[0]), 0), (np.sum(SL == u[1]), 1)])
    nm, im = min([(np.sum(SL == u[0]), 0), (np.sum(SL == u[1]), 1)])
    p = nm / len(SL)
    q = 1 - p
    YF = np.zeros_like(YT)
    for i in range(len(YT)):
        if ST[i] == u[im]:
            dist_best = float('inf')
            for t in np.linspace(np.min(YL), np.max(YL), 100):
                tmp1 = np.sum(YL[SL == u[iM]] < t) / nM
                tmp2 = np.sum(YL[SL == u[im]] < YT[i]) / nm
                dist = abs(tmp1 - tmp2)
                if dist < dist_best:
                    dist_best = dist
                    ts = t
            YF[i] = p * YT[i] + (1 - p) * ts
        else:
            dist_best = float('inf')
            for t in np.linspace(np.min(YL), np.max(YL), 100):
                tmp1 = np.sum(YL[SL == u[im]] < t) / nm
                tmp2 = np.sum(YL[SL == u[iM]] < YT[i]) / nM
                dist = abs(tmp1 - tmp2)
                if dist < dist_best:
                    dist_best = dist
                    ts = t
            YF[i] = q * YT[i] + (1 - q) * ts
    return YF
