import torch

from reliquary.validator import training


def test_histogram_quantile():
    hist = torch.zeros(training._LOGIT_HIST_BINS, dtype=torch.float64)
    hist[60] = 50   # maxima in [30.0, 30.5)
    hist[80] = 50   # maxima in [40.0, 40.5)
    assert training._histogram_quantile(hist, 0.5) == 30.5
    assert training._histogram_quantile(hist, 0.9) == 40.5


def test_logit_histogram_observes_max_logit_rows():
    training._reset_logit_histogram()
    lm_head = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.tensor([[10.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    hidden = torch.tensor([[3.1, 0, 0, 0], [4.05, 0, 0, 0]])
    training._hidden_logprob_block(hidden, torch.tensor([0, 1]), lm_head)
    metrics = training._logit_metrics()
    assert metrics["train/max_logit_p50"] == 31.5
    assert metrics["train/max_logit_p90"] == 41.0


def test_outside_clip_split_by_pi_old_probability():
    stats = training._new_kl_stats()
    ratio = torch.tensor([1.0, 2.0, 1.0, 0.5])
    old = torch.log(torch.tensor([0.01, 0.05, 0.5, 0.9]))
    zeros = torch.zeros(4)
    training._record_kl_stats(
        stats, ppo_tok=zeros, kl_tok=zeros, kl_log=zeros, scale_cat=zeros,
        ppo_ratio=ratio, ppo_log_ratio=torch.log(ratio),
        ppo_clip_active=torch.zeros(4, dtype=torch.bool),
        claimed_old=old, behavior_old=None, pi_old=old,
    )
    m = training._kl_telemetry_metrics(stats)
    assert m["train/ppo_outside_clip_ratio_p_lt_0_1"] == 0.5
    assert m["train/ppo_outside_clip_ratio_p_ge_0_1"] == 0.5
