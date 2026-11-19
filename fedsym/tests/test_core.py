import numpy as np
import pytest
import torch

from fedsym.align import aligned_merge, align_to, merge, rotate_output
from fedsym.losses import InfoNCE
from fedsym.models import SymBreakMLP, mlp_forward, parse_variant


def _layers(m):
    return [{k: v.detach() for k, v in ly.items()} for ly in m.eff_weights()]


def _permute(w, perms):
    w = [dict(ly) for ly in w]
    for l, p in enumerate(perms):
        for k in ("W", "b", "R", "a"):
            if k in w[l]:
                w[l][k] = w[l][k][p]
        w[l + 1]["W"] = w[l + 1]["W"][:, p]
        if "R" in w[l + 1]:
            w[l + 1]["R"] = w[l + 1]["R"][:, p]
    return w


def _scramble(w, g, shared=False):
    n = w[0]["W"].shape[0]
    p0 = torch.randperm(n, generator=g)
    perms = [p0 if shared else torch.randperm(w[l]["W"].shape[0], generator=g) for l in range(len(w) - 1)]
    w = _permute(w, perms)
    Q, _ = torch.linalg.qr(torch.randn(w[-1]["W"].shape[0], w[-1]["W"].shape[0], generator=g))
    return rotate_output(w, Q), Q


def test_symmetry_is_exact_and_alignment_recovers_it():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    m = SymBreakMLP(20, (64, 64), (64,), 16)
    w = _layers(m)
    ws, Q = _scramble(w, g)
    x = torch.randn(2048, 20)
    _, z = mlp_forward(w, x, 2)
    _, z_s = mlp_forward(ws, x, 2)
    loss = InfoNCE(16)
    z2 = z + 0.1 * torch.randn_like(z)
    assert torch.allclose(loss(z, z2), loss(z_s, z2 @ Q), atol=1e-5)
    for osb in ("diag", "mrl"):
        lf = InfoNCE(16, osb=osb, mrl_dims=(4, 8, 16))
        assert abs(lf(z, z2) - lf(z_s, z2 @ Q)) > 1e-3
    wa, R, st = align_to(ws, w, x)
    wa = rotate_output(wa, R)
    for A, B in zip(wa, w):
        assert torch.allclose(A["W"], B["W"], atol=1e-4) and torch.allclose(A["b"], B["b"], atol=1e-4)
    assert st["rot_explained"] > 0.999
    mn, ma = merge([w, ws], [1, 1]), merge([w, wa], [1, 1])
    assert not torch.allclose(mlp_forward(mn, x, 2)[1], z, atol=1e-2)
    assert torch.allclose(mlp_forward(ma, x, 2)[1], z, atol=1e-4)


@pytest.mark.parametrize("mech", ["res_id", "res_diag", "res_rand", "act_asym"])
def test_new_mechanisms_permute_consistently(mech):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(2)
    m = SymBreakMLP(20, (64, 64), (64,), 16, mechs=[mech])
    w = _layers(m)
    assert ("R" in w[1]) == mech.startswith("res") and ("a" in w[0]) == (mech == "act_asym")
    ws, Q = _scramble(w, g)
    x = torch.randn(4096, 20)
    _, z = mlp_forward(w, x, 2)
    assert torch.allclose(mlp_forward(ws, x, 2)[1], z @ Q, atol=1e-4)
    wa, R, _ = align_to(ws, w, x)
    wa = rotate_output(wa, R)
    assert torch.allclose(mlp_forward(wa, x, 2)[1], z, atol=1e-3)


def test_residual_symmetry_structure():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(512, 20)
    for mech, broken in (("res_id", False), ("res_diag", True), ("res_rand", True)):
        m = SymBreakMLP(20, (64, 64), (64,), 16, mechs=[mech])
        w = _layers(m)
        p = torch.randperm(64, generator=g)
        wp = _permute(w, [p, p, p])
        for ly, ly0 in zip(wp, w):
            if "R" in ly0:
                ly["R"] = ly0["R"]
        _, z = mlp_forward(w, x, 2)
        _, zp = mlp_forward(wp, x, 2)
        assert (not torch.allclose(z, zp, atol=1e-4)) == broken, mech


def test_variants_structure():
    m = SymBreakMLP(20, (64, 64), (64,), 16, mechs=["wasym"], arch_seed=3)
    m2 = SymBreakMLP(20, (64, 64), (64,), 16, mechs=["wasym"], arch_seed=3)
    assert torch.equal(m.fixed0, m2.fixed0) and torch.equal(m.mask0, m2.mask0)
    W_eff = m.eff_weights()[0]["W"]
    assert torch.equal(W_eff[m.mask0.bool()], m.fixed0[m.mask0.bool()])
    assert m.n_trainable() < sum(p.numel() for p in m.parameters())
    d = SymBreakMLP(20, (64,), (64,), 16, mechs=["dropasym"], drop_mean=0.1)
    assert abs(d.drop0.mean().item() - 0.1) < 1e-6 and d.drop0.std() > 0
    lr = SymBreakMLP(20, (64,), (64,), 16, mechs=["lrasym"])
    assert abs(lr.lrm0.log().mean().item()) < 1e-5
    assert parse_variant("wasym+res_diag+mrl") == (["wasym", "res_diag"], "mrl")
    assert parse_variant("wasym_mrl") == (["wasym"], "mrl")
    with pytest.raises(AssertionError):
        parse_variant("wasym+wasym_band")
    for mech in ("wasym", "wasym_band", "res_diag", "res_rand", "act_asym", "dropasym", "lrasym"):
        sig = SymBreakMLP(20, (64, 64), (64,), 16, mechs=[mech]).unit_signatures()
        n_cov = sum(len(s) > 0 for s in sig)
        assert n_cov == (2 if mech.startswith("res") else 3), mech


def test_band_prior_is_local():
    m = SymBreakMLP(64, (128, 128), (128,), 16, mechs=["wasym_band"])
    F = m.fixed1
    sim = torch.nn.functional.normalize(F, dim=1) @ torch.nn.functional.normalize(F, dim=1).T
    assert sim[10, 11] > 0.8 and sim[10, 74].abs() < 0.3


def test_wfixsym_control_is_permutation_symmetric():
    torch.manual_seed(0)
    m = SymBreakMLP(20, (64, 64), (64,), 16, mechs=["wfixsym"], wasym_scale=4.0, arch_seed=1)
    W0 = m.eff_weights()[0]["W"].detach()
    assert torch.equal(m.fixed0[0], m.fixed0[5]) and torch.equal(m.mask0[0], m.mask0[5])
    p = torch.randperm(64)
    with torch.no_grad():
        m.weights[0].copy_(m.weights[0][p])
    assert torch.equal(m.eff_weights()[0]["W"].detach(), W0[p])


def test_locality_metrics_and_restricted_alignment():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(4)
    m = SymBreakMLP(20, (64, 64), (64,), 16, mechs=["act_asym"], act_max=0.5)
    w, sig = _layers(m), m.unit_signatures()
    x = torch.randn(4096, 20)
    local = []
    for l in range(3):
        p = torch.arange(64)
        for i in range(0, 64, 2):
            if torch.rand(1, generator=g) < 0.5:
                p[i], p[i + 1] = p[i + 1].clone(), p[i].clone()
        local.append(p)
    ws = _permute(w, local)
    wa, _, st = align_to(ws, w, x, sig=sig, restrict_k=4)
    assert np.mean(st["loc_ratio"]) < 0.05 and min(st["loc_cov8"]) == 1.0
    _, z = mlp_forward(w, x, 2)
    assert torch.allclose(mlp_forward(wa, x, 2)[1], z, atol=1e-3)
    wr = _permute(w, [torch.randperm(64, generator=g) for _ in range(3)])
    _, _, st_r = align_to(wr, w, x, sig=sig)
    assert np.mean(st_r["loc_ratio"]) > 0.5
    merged, _ = aligned_merge([w, ws], [1, 1], x, sig=sig, restrict_k=4)
    assert torch.allclose(mlp_forward(merged, x, 2)[1], z, atol=1e-3)


@pytest.mark.parametrize('topology', ['random', 'ring'])
@pytest.mark.parametrize('K,g', [(10, 2), (10, 5), (10, 10), (10, 3)])
def test_gossip_groups_partition_clients(topology, K, g):
    from train import gossip_groups
    rng = np.random.default_rng(0)
    for r in range(4):
        groups = gossip_groups(K, g, r, topology, rng)
        flat = sorted(k for G in groups for k in G)
        assert flat == list(range(K))
        assert all(1 <= len(G) <= g for G in groups)
    if topology == 'ring' and g == 2:
        assert gossip_groups(K, g, 0, topology, rng) != gossip_groups(K, g, 1, topology, rng)
