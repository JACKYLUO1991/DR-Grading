import numpy as np
import math
from scipy import stats

if __name__ == '__main__':
    a = np.array([0.9913, 0.9627, 0.9607, 0.9369, 0.9636, 0.9866, 0.9649, 0.9795, 0.9383, 0.9777])
    m = a.mean()
    s = a.std() / math.sqrt(len(a))
    ci = 2.262 * s
    lower_bound = m - ci
    upper_bound = m + ci
    print(lower_bound, upper_bound)

    # auc = 0.966
    # auc_std = 0.0184
    # alpha = .95
    # lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    # ci = stats.norm.ppf(
    #     lower_upper_q,
    #     loc=auc,
    #     scale=auc_std)

    # ci[ci > 1] = 1
    # print('95% AUC CI:', ci)

    # import scipy.stats

    # def mean_confidence_interval(data, confidence=0.95):
    #     a = 1.0 * np.array(data)
    #     n = len(a)
    #     m, se = np.mean(a), scipy.stats.sem(a)
    #     h = se * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    #     return m, m - h, m + h

    # result = mean_confidence_interval(a, confidence=0.95)
    # print(result)

    import statsmodels.stats.api as sms

    conf = sms.DescrStatsW(a).tconfint_mean(0.05)
    print(conf)
