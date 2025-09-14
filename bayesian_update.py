import numpy as np

def bayesian_update(prior, likelihood):
    """
    贝叶斯在线更新：计算当前后验风险概率
    :param prior: P(E|H_{t-1}) 历史后验概率
    :param likelihood: P(E|T_t) 当前流量外泄似然
    :return: P(E|H_t) 当前后验概率
    """
    if prior < 0 or prior > 1 or likelihood < 0 or likelihood > 1:
        raise ValueError("Probabilities must be in [0, 1]")
    
    numerator = likelihood * prior
    denominator = likelihood * prior + (1 - likelihood) * (1 - prior)
    posterior = numerator / denominator
    return posterior

# 示例：模拟5次调用
calls = [0.65, 0.70, 0.50, 0.80, 0.75]  # 每次调用的P(E|T_t)
prior = 0.05  # 初始先验（低风险假设）

print("调用\tP(E|T_t)\t后验风险")
for i, p_et in enumerate(calls):
    posterior = bayesian_update(prior, p_et)
    print(f"{i+1}\t{p_et:.2f}\t\t{posterior:.3f}")
    prior = posterior  # 更新为下一次先验
