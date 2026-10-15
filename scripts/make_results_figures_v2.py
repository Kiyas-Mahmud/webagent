"""Results-section figures v2: two agents (Browser Use, SeeAct), each with and without WebFARM.

Usage:  .venv/bin/python scripts/make_results_figures_v2.py [R03 R04 ...]   (default: all)
Output: docs/figures_v2/<name>.pdf (vector, for the paper), .svg (editable vector), .png (preview, 300 dpi).

Data: web-v3 (A = Browser Use, C = Browser Use + WebFARM), web-v4-seeact (S = SeeAct),
web-v5-seeact-ours (T = SeeAct + WebFARM), Task 1 pilot records and labels. Every plotted
number is recomputed from the raw run files and asserted against the values reported in
docs/Results_Comparison_Handout.pdf (Tables 2, 6-20), so figures and tables cannot disagree.

Visual concept (fixed across all figures):
  colour  = state: grey = agent alone, blue = agent + WebFARM, red = harm only;
  agent   = by panel or group label; where both agents share an axis: line style
            (solid Browser Use, dashed SeeAct) or marker shape (circle Browser Use, square SeeAct);
  text    = near-black only, 7-8.5 pt at final size, Liberation Sans (Arial-metric);
  sizes   = Elsevier artwork widths 90 / 140 / 190 mm.
"""
from __future__ import annotations

import collections
import functools
import inspect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_results_figures as v1  # noqa: E402  (shared style, palette, helpers and Task 1 figure)

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = v1.ROOT
CMP = ROOT / '.task2-assets/comparison'
RUNS = {'A': CMP / 'web-v3', 'C': CMP / 'web-v3', 'S': CMP / 'web-v4-seeact', 'T': CMP / 'web-v5-seeact-ours'}
OUT = ROOT / 'docs/figures_v2'
MM = v1.MM
BASE, WEBFARM, WEBFARM_LIGHT, HURT, SAME, INK, GRID, AXIS = v1.BASE, v1.OURS, v1.OURS_LIGHT, v1.HURT, v1.SAME, v1.INK, v1.GRID, v1.AXIS
plt.rcParams.update({'svg.fonttype': 'none', 'hatch.linewidth': 0.7})

LABEL = {'A': 'Browser Use', 'C': 'Browser Use + WebFARM', 'S': 'SeeAct', 'T': 'SeeAct + WebFARM'}
COLOR = {'A': BASE, 'C': WEBFARM, 'S': BASE, 'T': WEBFARM}
DASH = {'A': '-', 'C': '-', 'S': (0, (4, 2)), 'T': (0, (4, 2))}
MARK = {'A': 'o', 'C': 'o', 'S': 's', 'T': 's'}
AGENTS = (('Browser Use', 'A', 'C'), ('SeeAct', 'S', 'T'))
SPLITS = (('Development', 'development'), ('Held-out', 'held_out'), ('All', None))

# Documented values (Results_Comparison_Handout.pdf, Tables 6-20).
COMPLETED = {'A': 61, 'C': 72, 'S': 52, 'T': 69}
BY_SPLIT = {('A', 'development'): 28, ('C', 'development'): 35, ('S', 'development'): 24, ('T', 'development'): 31,
            ('A', 'held_out'): 33, ('C', 'held_out'): 37, ('S', 'held_out'): 28, ('T', 'held_out'): 38}
PAIRS = {('A', 'C', 'development'): (8, 1), ('A', 'C', 'held_out'): (5, 1), ('A', 'C', None): (13, 2),
         ('S', 'T', 'development'): (9, 2), ('S', 'T', 'held_out'): (12, 2), ('S', 'T', None): (21, 4)}
MEAN_SECONDS = {'A': 220, 'C': 175, 'S': 182, 'T': 123}


def check(name, got, want, tol=0.05):
    if abs(got - want) > tol:
        raise AssertionError(f'{name}: computed {got}, documented {want}')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02, **kw)
    plt.close(fig)
    print('wrote', OUT / name, '(.pdf .svg .png)')


def ptext(p):
    return f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'


def legend_patches(items, hatch=None):
    return [matplotlib.patches.Patch(facecolor=c, edgecolor='white', hatch=h, linewidth=0) for c, h in items]


def spread(values, gap):
    """Shift label positions apart (keeping order) so neighbours are at least `gap` apart."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for _ in range(50):
        moved = False
        for j in range(1, len(pos)):
            if pos[j] - pos[j - 1] < gap:
                mid = (pos[j] + pos[j - 1]) / 2
                pos[j - 1], pos[j] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for i, p in zip(order, pos):
        out[i] = p
    return out


# ---------------------------------------------------------------- data
@functools.lru_cache(maxsize=None)
def load4():
    plan = json.loads((RUNS['A'] / 'plan.json').read_text())
    eps = {}
    for e in plan['episodes']:
        if e['system'] != 'A':
            continue
        for s in 'ACST':
            p = RUNS[s] / 'episodes' / f"{e['repeat']}-{e['task']}-{s}" / 'result.json'
            eps[(e['repeat'], e['task'], s)] = dict(json.loads(p.read_text()), split=e['split'], folder=p.parent)
    assert len(eps) == 336, len(eps)
    for s, n in COMPLETED.items():
        assert sum(v['completion'] for k, v in eps.items() if k[2] == s) == n, s
    for (s, sp), n in BY_SPLIT.items():
        assert sum(v['completion'] for k, v in eps.items() if k[2] == s and v['split'] == sp) == n, (s, sp)
    for s, m in MEAN_SECONDS.items():
        vals = [v['elapsed_seconds'] for k, v in eps.items() if k[2] == s]
        check(f'mean seconds {s}', sum(vals) / len(vals), m, tol=0.5)
    return eps


def keys(split=None):
    eps = load4()
    return sorted({(r, t) for (r, t, s), v in eps.items() if s == 'A' and (split is None or v['split'] == split)})


def paired(raw, fw, split=None):
    eps = load4()
    return [(eps[k + (raw,)]['completion'], eps[k + (fw,)]['completion']) for k in keys(split)]


def helped_hurt(raw, fw, split=None):
    pr = paired(raw, fw, split)
    h, u = sum((not a) and b for a, b in pr), sum(a and not b for a, b in pr)
    assert (h, u) == PAIRS[(raw, fw, split)], (raw, fw, split, h, u)
    return h, u, len(pr) - h - u


def action_events(folder):
    return [json.loads(a.read_text()) for a in sorted(folder.glob('action-*.json'))]


def choices_with_steps(ep, system):
    """(choice record, step index of the action it led to) for every option prompt of a WebFARM episode."""
    d = ep['folder']
    if system == 'C':
        step_of = {}
        for e in action_events(d):
            sel = (e.get('native_selection') or {}).get('actor_request_id')
            if sel:
                step_of[sel] = e['step']
        return [(json.loads(c.read_text()), step_of.get(c.name.replace('-choice.json', ''))) for c in sorted(d.glob('actor-*-choice.json'))]
    # T: actor prefixes are numbered per call: a step is [choice picked] or [choice 0, turn 0] or [turn 0].
    items = sorted([(int(f.name[6:10]), 'choice', f) for f in d.glob('actor-*-choice.json')] +
                   [(int(f.name[6:10]), 'turn', f) for f in d.glob('actor-*-turn0.json')])
    step, out = 0, []
    for _, kind, f in items:
        if kind == 'choice':
            j = json.loads(f.read_text())
            out.append((j, step))
            if j['picked'] is not None:
                step += 1
        else:
            step += 1
    assert step == len(list(d.glob('action-*.json'))), (d, step)
    return out


def assessments(ep):
    return {int(a.stem[-4:]): json.loads(a.read_text()) for a in ep['folder'].glob('assessment-*.json')}


def recovery_trace(ep, system):
    ch = choices_with_steps(ep, system)
    ass = assessments(ep)
    return {'opened': bool(list(ep['folder'].glob('note-*.json'))), 'offered': bool(ch),
            'taken': any(c['picked'] is not None for c, _ in ch),
            'verified': any(a['assessment']['phase'] == 'recovery_assessment' and a['assessment']['signals']['outcome_label'] == 'SUCCESS'
                            for a in ass.values()),
            'completed': ep['completion']}


def option_kind(label):
    return v1.option_kind(label)


# ---------------------------------------------------------------- R01: Task 1, trained vs prompted (v1 figure, unchanged)
def r01_task1():
    def redirected(fig, _name):
        save(fig, 'R01_task1_trained_vs_prompted')
    v1.save, saved = redirected, v1.save
    try:
        v1.fig6_task1()
    finally:
        v1.save = saved


# ---------------------------------------------------------------- R02: Task 1, confusion matrices
def r02_confusion():
    labels = json.loads((ROOT / 'docs/evidence/task1-assessment-prepared-v1/scoring-only.json').read_text())
    recs = [json.loads(f.read_text()) for f in (ROOT / '.task1-assets/runs/task1-qwen25-dual-v1/pilot/qwen25_heads').glob('*.json')
            if not f.name.endswith('started.json')]
    recs = [r for r in recs if 'phase' in r]
    assert len(recs) == 360 and all(r['request_key'] in labels for r in recs)
    order = ('FAILURE', 'SUCCESS')
    cmap = LinearSegmentedColormap.from_list('webfarm_blues', ['#f2f6fc', '#9ec5f4', '#2a78d6', '#123f7a'])
    fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 62 * MM), gridspec_kw={'wspace': 0.55})
    for ax, (phase, title, want) in zip(axes, (('interaction_assessment', '(a) "Did this step fail?"', 0.649),
                                               ('recovery_assessment', '(b) "Did the recovery work?"', 0.841))):
        rs = [r for r in recs if r['phase'] == phase]
        cm = collections.Counter((labels[r['request_key']]['outcome_label'], r['predictions']['outcome_label']) for r in rs)
        tp, fn, fp, tn = cm[('FAILURE', 'FAILURE')], cm[('FAILURE', 'SUCCESS')], cm[('SUCCESS', 'FAILURE')], cm[('SUCCESS', 'SUCCESS')]
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        check(f'{phase} MCC', mcc, want, tol=0.0005)
        for i, truth in enumerate(order):
            row = sum(cm[(truth, p)] for p in order)
            for j, pred in enumerate(order):
                n = cm[(truth, pred)]
                share = n / row
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(share), edgecolor='white', linewidth=1.5))
                dark = share > 0.55
                ax.text(j + 0.5, i + 0.43, str(n), ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if dark else INK)
                ax.text(j + 0.5, i + 0.7, f'{100 * share:.0f} % of row', ha='center', va='center', fontsize=7,
                        color='white' if dark else INK)
        ax.set_xlim(0, 2); ax.set_ylim(2, 0)
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['Failure', 'Success'])
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['Failure', 'Success'])
        ax.set_xlabel('Predicted by WebFARM assessor'); ax.set_ylabel('Gold label')
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{title}\n{len(rs)} cases, MCC {mcc:.3f}', loc='left', color=INK, fontsize=8.5, linespacing=1.3)
    save(fig, 'R02_task1_confusion_matrices')


# ---------------------------------------------------------------- R03: main result, two agents
def r03_main():
    doc_p = {('A', 'C', 'development'): 0.039, ('A', 'C', 'held_out'): 0.219, ('A', 'C', None): 0.007,
             ('S', 'T', 'development'): 0.065, ('S', 'T', 'held_out'): 0.013, ('S', 'T', None): 0.0009}
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 74 * MM), sharey=True, gridspec_kw={'wspace': 0.06})
    width, gap = 0.34, 0.02
    for ax, (agent, raw, fw), letter in zip(axes, AGENTS, 'ab'):
        for i, (name, sp) in enumerate(SPLITS):
            pr = paired(raw, fw, sp)
            n = len(pr)
            a, c = sum(x for x, _ in pr), sum(y for _, y in pr)
            h, u, _ = helped_hurt(raw, fw, sp)
            pa, la, ua = v1.wilson(a, n)
            pc, lc, uc = v1.wilson(c, n)
            p = v1.sign_p(h, u)
            check(f'{agent} {name} p', p, doc_p[(raw, fw, sp)], tol=0.0006)
            for x, val, lo, hi, color in ((i - width / 2 - gap / 2, pa, la, ua, BASE), (i + width / 2 + gap / 2, pc, lc, uc, WEBFARM)):
                ax.bar(x, val, width, color=color, zorder=2)
                ax.plot([x, x], [lo, hi], color=INK, lw=0.7, zorder=3)
                for yy in (lo, hi):
                    ax.plot([x - 0.05, x + 0.05], [yy, yy], color=INK, lw=0.7, zorder=3)
                ax.text(x, 4, f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='white', fontweight='bold', zorder=4)
            top = max(ua, uc) + 4
            x0, x1 = i - width / 2 - gap / 2, i + width / 2 + gap / 2
            ax.plot([x0, x0, x1, x1], [top - 1.5, top, top, top - 1.5], color=INK, lw=0.6)
            ax.text(i, top + 1.2, f'+{pc - pa:.1f} pts\n{ptext(p)}', ha='center', va='bottom', fontsize=7.5, color=INK,
                    linespacing=1.15)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f'{name}\n({len(paired(raw, fw, sp))} pairs)' for name, sp in SPLITS])
        ax.tick_params(axis='x', length=0, pad=4)
        ax.set_xlim(-0.6, 2.6)
        ax.grid(axis='y', zorder=0)
        ax.set_title(f'({letter}) {agent}', loc='left', color=INK, fontweight='bold')
    axes[0].set_ylim(0, 120)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(range(10, 100, 20)))
    axes[0].spines['left'].set_bounds(0, 100)
    axes[0].set_ylabel('Task completion (%)')
    axes[1].spines['left'].set_visible(False)
    axes[1].tick_params(axis='y', which='both', length=0)
    fig.legend(legend_patches([(BASE, None), (WEBFARM, None)]), ['Agent alone', 'Agent + WebFARM'], loc='upper center',
               bbox_to_anchor=(0.5, 1.03), ncol=2, handlelength=1.2, columnspacing=1.8)
    save(fig, 'R03_main_result_two_agents')


# ---------------------------------------------------------------- R04: transfer slope chart
def r04_slope():
    eps = load4()
    fig, axes = plt.subplots(1, 3, figsize=(140 * MM, 64 * MM), sharey=True, gridspec_kw={'wspace': 0.12})
    for ax, (name, sp) in zip(axes, SPLITS):
        ks = keys(sp)
        rate = {s: 100 * sum(eps[k + (s,)]['completion'] for k in ks) / len(ks) for s in 'ACST'}
        for agent, raw, fw in AGENTS:
            ax.plot([0, 1], [rate[raw], rate[fw]], color=INK, lw=1.0, ls=DASH[raw], zorder=2)
            ax.plot(0, rate[raw], MARK[raw], markersize=6, color=BASE, markeredgecolor='white', markeredgewidth=0.7, zorder=3)
            ax.plot(1, rate[fw], MARK[fw], markersize=6, color=WEBFARM, markeredgecolor='white', markeredgewidth=0.7, zorder=3)
        left = spread([rate['A'], rate['S']], 4.2)
        right = spread([rate['C'], rate['T']], 4.2)
        for y, s in zip(left, 'AS'):
            ax.text(-0.09, y, f'{rate[s]:.1f}', ha='right', va='center', fontsize=7.5, color=INK)
        for y, s in zip(right, 'CT'):
            ax.text(1.09, y, f'{rate[s]:.1f}', ha='left', va='center', fontsize=7.5, color=INK, fontweight='bold')
        g0, g1 = rate['A'] - rate['S'], rate['C'] - rate['T']
        ax.text(0.5, 47.5, f'gap between agents\n{g0:.1f} → {g1:.1f} pts'.replace('-', '−'), ha='center', va='bottom',
                fontsize=7, color=INK, linespacing=1.25)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['alone', '+ WebFARM'])
        ax.tick_params(axis='x', length=0)
        ax.spines['bottom'].set_visible(False)
        ax.grid(axis='y', zorder=0)
        ax.set_title(f'{name} ({len(ks)} pairs)', color=INK, fontsize=8)
    axes[0].set_ylim(45, 95)
    axes[0].set_yticks(range(50, 91, 10))
    axes[0].set_ylabel('Task completion (%)')
    for ax in axes[1:]:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    handles = [plt.Line2D([], [], color=INK, lw=1.0, ls=DASH['A'], marker='o', markersize=5, markerfacecolor=BASE, markeredgecolor='white'),
               plt.Line2D([], [], color=INK, lw=1.0, ls=DASH['S'], marker='s', markersize=5, markerfacecolor=BASE, markeredgecolor='white')]
    fig.legend(handles, ['Browser Use', 'SeeAct'], loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=2,
               handlelength=2.6, columnspacing=2.0)
    save(fig, 'R04_transfer_slope')


# ---------------------------------------------------------------- R05: paired outcomes, diverging
def r05_paired():
    rows = []
    for agent, raw, fw in AGENTS:
        for name, sp in SPLITS:
            h, u, same = helped_hurt(raw, fw, sp)
            pr = paired(raw, fw, sp)
            fails, oks = sum(not a for a, _ in pr), sum(a for a, _ in pr)
            rows.append((agent, name, h, u, same, fails, oks))
    fig, ax = plt.subplots(figsize=(190 * MM, 66 * MM))
    fig.subplots_adjust(left=0.2, right=0.6)
    ys = []
    y = 0.0
    for i, r in enumerate(rows):
        if i == 3:
            y += 0.7
        ys.append(y)
        y += 1
    for y, (agent, name, h, u, same, fails, oks) in zip(ys, rows):
        bold = name == 'All'
        ax.barh(y, h, height=0.62, color=WEBFARM, zorder=2)
        ax.barh(y, -u, height=0.62, color=HURT, zorder=2)
        ax.text(h + 0.4, y, str(h), ha='left', va='center', fontsize=7.5, color=INK, fontweight='bold' if bold else 'normal')
        ax.text(-u - 0.4, y, str(u), ha='right', va='center', fontsize=7.5, color=INK, fontweight='bold' if bold else 'normal')
        ax.text(1.03, y, f'{100 * h / fails:.0f} % ({h}/{fails})', transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=7.5, color=INK, fontweight='bold' if bold else 'normal')
        ax.text(1.33, y, f'{100 * u / oks:.0f} % ({u}/{oks})', transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=7.5, color=INK, fontweight='bold' if bold else 'normal')
        ax.text(1.63, y, str(same), transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7.5, color=INK)
    for x, head in ((1.03, 'Rescue rate\n(failures fixed)'), (1.33, 'Harm rate\n(successes lost)'), (1.63, 'Same\noutcome')):
        ax.text(x, -0.75, head, transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=7, color=INK,
                fontweight='bold', linespacing=1.15)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{a}: {n}' for a, n, *_ in rows], fontsize=7.5)
    for lab, (_, n, *_) in zip(ax.get_yticklabels(), rows):
        if n == 'All':
            lab.set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(ys[-1] + 0.6, -0.8)
    ax.axvline(0, color=INK, lw=0.7, zorder=3)
    ax.set_xlim(-6, 24)
    ax.set_xticks([-5, 0, 5, 10, 15, 20])
    ax.set_xticklabels(['5', '0', '5', '10', '15', '20'])
    ax.spines['bottom'].set_bounds(-5, 20)
    ax.set_xlabel('Paired episodes changed by WebFARM\n← hurt (agent alone completed, + WebFARM failed)     helped (the reverse) →',
                  linespacing=1.4)
    ax.grid(axis='x', zorder=0)
    save(fig, 'R05_paired_outcomes')


# ---------------------------------------------------------------- R06: cost
def r06_cost():
    eps = load4()
    order = ('A', 'C', 'S', 'T')
    pos = {'A': 0, 'C': 1, 'S': 2.6, 'T': 3.6}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(190 * MM, 66 * MM), gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.3})
    for s in order:
        vals = [v['elapsed_seconds'] for k, v in eps.items() if k[2] == s]
        b = ax1.boxplot([vals], positions=[pos[s]], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', lw=1.4), whiskerprops=dict(color=INK, lw=0.7),
                        capprops=dict(color=INK, lw=0.7), boxprops=dict(lw=0))
        b['boxes'][0].set_facecolor(COLOR[s])
        b['boxes'][0].set_zorder(2)
        mean = sum(vals) / len(vals)
        ax1.plot(pos[s], mean, 'D', markersize=4.2, color='white', markeredgecolor=INK, markeredgewidth=0.7, zorder=4)
        ax1.text(pos[s] + 0.33, mean, f'{mean:.0f} s', ha='left', va='center', fontsize=7.5, color=INK, fontweight='bold')
    ax1.set_xticks([pos[s] for s in order])
    ax1.set_xticklabels(['alone', '+ WebFARM', 'alone', '+ WebFARM'])
    for x, agent in ((0.5, 'Browser Use'), (3.1, 'SeeAct')):
        ax1.text(x, -0.12, agent, transform=ax1.get_xaxis_transform(), ha='center', va='top', fontsize=8, color=INK, fontweight='bold')
    ax1.tick_params(axis='x', length=0)
    ax1.set_ylabel('Wall-clock time per episode (s)')
    ax1.set_ylim(0, None)
    ax1.grid(axis='y', zorder=0)
    ax1.set_xlim(-0.6, 4.4)
    ax1.set_title('(a) Time per episode (84 episodes each)', loc='left', color=INK)
    # (b) steps on pairs both variants completed
    w = 0.36
    for i, (agent, raw, fw) in enumerate(AGENTS):
        ks = [k for k in keys() if eps[k + (raw,)]['completion'] and eps[k + (fw,)]['completion']]
        for off, s in ((-w / 2 - 0.02, raw), (w / 2 + 0.02, fw)):
            m = sum(eps[k + (s,)]['agent_steps'] for k in ks) / len(ks)
            want = {'A': 3.08, 'C': 3.32, 'S': 3.31, 'T': 3.40}[s]
            check(f'steps {s}', m, want, tol=0.006)
            ax2.bar(i + off, m, w, color=COLOR[s], zorder=2)
            ax2.text(i + off, m + 0.08, f'{m:.2f}', ha='center', va='bottom', fontsize=7.5, color=INK, fontweight='bold')
        ax2.text(i, -0.05, f'{agent}\n({len(ks)} pairs)', transform=ax2.get_xaxis_transform(), ha='center', va='top',
                 fontsize=8, color=INK, linespacing=1.25)
    ax2.set_xticks([])
    ax2.set_ylim(0, 4.2)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_ylabel('Mean steps per episode')
    ax2.grid(axis='y', zorder=0)
    ax2.set_xlim(-0.6, 1.6)
    ax2.spines['bottom'].set_visible(False)
    ax2.set_title('(b) Steps on tasks both variants completed', loc='left', color=INK)
    fig.legend(legend_patches([(BASE, None), (WEBFARM, None)]), ['Agent alone', 'Agent + WebFARM'], loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=2, handlelength=1.2, columnspacing=1.8)
    save(fig, 'R06_cost')


# ---------------------------------------------------------------- R07: step budget
def r07_step_budget():
    eps = load4()
    budget = 15
    fig, ax = plt.subplots(figsize=(140 * MM, 70 * MM))
    fig.subplots_adjust(right=0.72)
    finals = {}
    for s in 'ACST':
        done_at = [v['agent_steps'] for k, v in eps.items() if k[2] == s and v['completion']]
        assert max(done_at) <= budget
        ks = list(range(0, budget + 1))
        ys = [100 * sum(d <= k for d in done_at) / 84 for k in ks]
        ax.step(ks, ys, where='post', color=COLOR[s], lw=1.7, ls=DASH[s], zorder=3 if s in 'CT' else 2)
        finals[s] = ys[-1]
    for s, want in (('A', 72.6), ('C', 85.7), ('S', 61.9), ('T', 82.1)):
        check(f'final {s}', finals[s], want)
    ylab = spread([finals[s] for s in 'ACST'], 5.2)
    for y, s in zip(ylab, 'ACST'):
        ax.annotate(f'{LABEL[s]}  {finals[s]:.1f} %', xy=(budget, finals[s]), xytext=(budget + 0.8, y), ha='left', va='center',
                    fontsize=7.5, color=INK, fontweight='bold' if s in 'CT' else 'normal',
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.6, shrinkA=0, shrinkB=1), annotation_clip=False)
    ax.set_xlim(0, budget)
    ax.set_xticks(range(0, budget + 1))
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel('Step budget (maximum actions allowed per episode)')
    ax.set_ylabel('Tasks completed within budget (%)')
    ax.grid(axis='y', zorder=0)
    handles = [plt.Line2D([], [], color=INK, lw=1.4, ls=DASH['A']), plt.Line2D([], [], color=INK, lw=1.4, ls=DASH['S']),
               plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=WEBFARM)]
    ax.legend(handles, ['Browser Use', 'SeeAct', 'alone', '+ WebFARM'], loc='lower right', ncol=2, handlelength=2.2,
              columnspacing=1.4, bbox_to_anchor=(1.0, 0.04))
    save(fig, 'R07_step_budget')


# ---------------------------------------------------------------- R08: which recovery options worked, both agents
KINDS = ['Open a goal-matching link', 'Search the site', 'Press Enter (after a search)', 'Scroll to top, then search',
         'Press Enter to submit a field', 'Scroll back to top', 'Go back', 'Decided itself (answered "0")']


def option_stats(system):
    eps = load4()
    offered, verdict, prompts = collections.Counter(), collections.Counter(), 0
    for k, v in eps.items():
        if k[2] != system:
            continue
        ass = assessments(v)
        for c, step in choices_with_steps(v, system):
            prompts += 1
            for o in c['options']:
                offered[option_kind(o['label'])] += 1
            kind = KINDS[-1] if c['picked'] is None else option_kind(c['picked']['label'])
            a = ass.get(step)
            verdict[(kind, a['assessment']['signals']['outcome_label'] if a else 'none')] += 1
    return offered, verdict, prompts


def r08_options():
    stats = {s: option_stats(s) for s in 'CT'}
    taken = {s: sum(n for (k, _), n in stats[s][1].items() if k != KINDS[-1]) for s in 'CT'}
    assert (stats['C'][2], taken['C']) == (86, 69) and (stats['T'][2], taken['T']) == (155, 143), (stats['C'][2], taken, stats['T'][2])
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 78 * MM), sharey=True, gridspec_kw={'wspace': 0.42})
    fig.subplots_adjust(left=0.2, right=0.9)
    xmax = 60
    for ax, s, letter in zip(axes, 'CT', 'ab'):
        offered, verdict, prompts = stats[s]
        for y, k in enumerate(KINDS[::-1]):
            left = 0
            for key, color, txt in (('SUCCESS', WEBFARM, 'white'), ('FAILURE', HURT, 'white'), ('none', SAME, INK)):
                n = verdict.get((k, key), 0)
                if n:
                    ax.barh(y, n, height=0.62, left=left, color=color, edgecolor='white', linewidth=0.8, zorder=2)
                    if n >= 3:
                        ax.text(left + n / 2, y, str(n), ha='center', va='center', fontsize=7, color=txt, fontweight='bold')
                    left += n
            chosen = sum(verdict.get((k, key), 0) for key in ('SUCCESS', 'FAILURE', 'none'))
            right = f'{chosen}/{prompts}' if k == KINDS[-1] else f'{chosen}/{offered[k]}'
            ax.text(1.02, y, right, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7.5, color=INK)
        ax.text(1.02, len(KINDS) - 0.4, 'chosen/\noffered', transform=ax.get_yaxis_transform(), ha='left', va='bottom',
                fontsize=7, color=INK, fontweight='bold', linespacing=1.1)
        ax.set_xlim(0, xmax)
        ax.set_xticks(range(0, xmax + 1, 10))
        ax.set_xlabel('Times chosen by the actor')
        ax.grid(axis='x', zorder=0)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_title(f'({letter}) WebFARM on {LABEL[s].split(" +")[0]}: {prompts} option prompts', loc='left', color=INK, pad=16)
    axes[0].set_yticks(range(len(KINDS)))
    axes[0].set_yticklabels(KINDS[::-1], fontsize=7.5)
    fig.legend(legend_patches([(WEBFARM, None), (HURT, None), (SAME, None)]),
               ['Next step judged successful by P1', 'Next step judged failed by P1', 'No P1 judgement (episode ended)'],
               loc='upper center', bbox_to_anchor=(0.55, 1.11), ncol=3, handlelength=1.2, columnspacing=1.6)
    save(fig, 'R08_recovery_options_both_agents')


# ---------------------------------------------------------------- R09: failure modes of the two agents
ACT_TYPES = [('SCROLL', 'Scroll', '#eb6834', INK), ('CLICK', 'Click / select / hover', '#1baf7a', INK),
             ('TYPE', 'Type', '#eda100', INK), ('NAVIGATE', 'Navigate / back', '#e87ba4', INK),
             ('PRESS_KEY', 'Press key', '#008300', 'white'), ('NONE', 'No action, done or invalid output', '#4a3aa7', 'white')]


def act_kind(e):
    t = e.get('action_type')
    if t in ('SELECT', 'HOVER'):
        return 'CLICK'
    if t in (None, 'NONE', 'DONE'):
        return 'NONE'
    return t


def longest_run(folder):
    best = run = 0
    prev = None
    for e in action_events(folder):
        t = act_kind(e)
        if t == 'NONE':
            prev, run = None, 0
            continue
        if t == 'SCROLL':
            up = e.get('seeact_action') in ('SCROLL UP', 'PRESS HOME', 'PRESS PAGEUP') or '"down": false' in str(e.get('value') or '')
            sig = ('SCROLL', up)
        else:
            el = e.get('element')
            sig = (t, json.dumps(el, sort_keys=True) if el else e.get('target'),
                   str(e.get('value') or '') if t in ('TYPE', 'NAVIGATE', 'PRESS_KEY') else '')
        run = run + 1 if sig == prev else 1
        prev = sig
        best = max(best, run)
    return best


def r09_failure_modes():
    eps = load4()
    order = ('A', 'C', 'S', 'T')
    mix, nep = {}, {}
    for s in order:
        c = collections.Counter()
        n = 0
        for k, v in eps.items():
            if k[2] == s and not v['completion']:
                n += 1
                c.update(act_kind(e) for e in action_events(v['folder']))
        mix[s], nep[s] = c, n
    assert {s: nep[s] for s in order} == {'A': 23, 'C': 12, 'S': 32, 'T': 15}
    loops = {s: [(longest_run(v['folder']), v['completion']) for k, v in eps.items() if k[2] == s] for s in order}
    n_loop = {s: sum(b >= 5 for b, _ in loops[s]) for s in order}
    n_loop_fail = {s: sum(b >= 5 and not ok for b, ok in loops[s]) for s in order}
    assert (n_loop['A'], n_loop_fail['A'], n_loop['C'], n_loop_fail['C']) == (16, 15, 7, 3), n_loop
    assert n_loop_fail['S'] == 19, n_loop_fail
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(190 * MM, 64 * MM), gridspec_kw={'width_ratios': [1.6, 1], 'wspace': 0.32})
    # (a) action mix of failed episodes
    rows = list(order)[::-1]
    for y, s in enumerate(rows):
        total = sum(mix[s].values())
        left = 0
        for key, _, color, txt in ACT_TYPES:
            share = 100 * mix[s].get(key, 0) / total
            if share:
                ax1.barh(y, share, height=0.62, left=left, color=color, edgecolor='white', linewidth=0.8, zorder=2)
                if share >= 7:
                    ax1.text(left + share / 2, y, f'{share:.0f}%', ha='center', va='center', fontsize=7, color=txt, fontweight='bold')
                left += share
    if True:
        a_scroll = 100 * mix['A']['SCROLL'] / sum(mix['A'].values())
        s_click = 100 * mix['S']['CLICK'] / sum(mix['S'].values())
        assert a_scroll > 60 and s_click > 80, (a_scroll, s_click)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([f'{LABEL[s]}\n({nep[s]} failed episodes)' for s in rows], fontsize=7.5, linespacing=1.15)
    ax1.tick_params(axis='y', length=0)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlim(0, 100)
    ax1.set_xticks(range(0, 101, 20))
    ax1.set_xlabel('Share of actions in failed episodes (%)')
    ax1.set_title('(a) What the agent did in the episodes it failed', loc='left', color=INK, pad=26)
    ax1.legend(legend_patches([(t[2], None) for t in ACT_TYPES]), [t[1] for t in ACT_TYPES], loc='upper center',
               bbox_to_anchor=(0.42, 1.2), ncol=3, handlelength=1.1, columnspacing=1.0, fontsize=7)
    # (b) loops of 5+ identical actions
    pos = {'A': 0, 'C': 1.15, 'S': 2.9, 'T': 4.05}
    for s in order:
        ax2.bar(pos[s], n_loop[s], 0.75, color=COLOR[s], zorder=2)
        ax2.text(pos[s], n_loop[s] + 0.6, f'{n_loop[s]}\n({n_loop_fail[s]} failed)', ha='center', va='bottom', fontsize=7,
                 color=INK, linespacing=1.15)
    ax2.set_xticks([pos[s] for s in order])
    ax2.set_xticklabels(['alone', '+ Web-\nFARM', 'alone', '+ Web-\nFARM'], fontsize=7, linespacing=1.1)
    for x, agent in ((0.575, 'Browser Use'), (3.475, 'SeeAct')):
        ax2.text(x, -0.2, agent, transform=ax2.get_xaxis_transform(), ha='center', va='top', fontsize=8, color=INK, fontweight='bold')
    ax2.tick_params(axis='x', length=0)
    ax2.set_ylim(0, max(n_loop.values()) + 8)
    ax2.set_ylabel('Episodes (of 84)')
    ax2.grid(axis='y', zorder=0)
    ax2.set_xlim(-0.6, 4.65)
    ax2.set_title('(b) Episodes with a loop of\n5+ identical actions', loc='left', color=INK, pad=6, linespacing=1.2)
    save(fig, 'R09_failure_modes')


# ---------------------------------------------------------------- R10: detector scores on both agents
def r10_detector():
    eps = load4()
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), gridspec_kw={'wspace': 0.18})
    bins = [i / 20 for i in range(21)]
    results = {}
    for ax, s, letter in zip(axes, 'CT', 'ab'):
        ps = []
        for k, v in eps.items():
            if k[2] != s:
                continue
            for j in assessments(v).values():
                a = j['assessment']
                if a['phase'] == 'interaction_assessment' and not j['transition']['terminated']:
                    ps.append(a['outcome_probabilities']['FAILURE'])
        n, hi = len(ps), sum(p >= 0.9 for p in ps)
        gated, low = sum(0.5 < p < 0.9 for p in ps), sum(p <= 0.5 for p in ps)
        results[s] = (n, hi, gated)
        counts = [sum(lo <= p < h or (h == 1.0 and p == 1.0) for p in ps) for lo, h in zip(bins[:-1], bins[1:])]
        for lo, c in zip(bins[:-1], counts):
            color = WEBFARM if lo >= 0.9 else ('#a9a8a2' if lo >= 0.5 else '#d6d5cf')
            ax.bar(lo + 0.025, c, 0.046, color=color, zorder=2)
        ymax = max(counts)
        ax.plot([0.9, 0.9], [0, ymax * 1.12], color=INK, lw=0.9, zorder=3)
        ax.text(0.9, ymax * 1.14, 'gate 0.9', ha='center', va='bottom', fontsize=7.5, color=INK, fontweight='bold')
        ax.set_ylim(0, ymax * 1.85)
        ax.legend(legend_patches([('#d6d5cf', None), ('#a9a8a2', None), (WEBFARM, None)]),
                  [f'{low} steps judged successful (P \u2264 0.5)', f'{gated} uncertain alarms (0.5 < P < 0.9), held back by the gate',
                   f'{hi} confident failures (P \u2265 0.9): recovery opened'], loc='upper left', bbox_to_anchor=(0.0, 1.0),
                  fontsize=7, handlelength=1.1, labelspacing=0.35)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
        ax.set_xlabel('P1 probability that the step failed, P(failure)')
        ax.grid(axis='y', zorder=0)
        ax.set_title(f'({letter}) WebFARM on {LABEL[s].split(" +")[0]}: {n} non-terminal steps', loc='left', color=INK)
    axes[0].set_ylabel('Number of steps')
    assert results['C'][:2] == (244, 42), results['C']
    save(fig, 'R10_detector_scores_both_agents')


# ---------------------------------------------------------------- R11: intervention timeline
def timeline_rows(system):
    eps = load4()
    rows = []
    for k, v in eps.items():
        if k[2] != system:
            continue
        notes = sorted(int(f.stem[-4:]) for f in v['folder'].glob('note-*.json'))
        if not notes:
            continue
        options = sorted(step for c, step in choices_with_steps(v, system) if c['picked'] is not None and step is not None)
        rows.append(dict(first=notes[0] + 1, alarms=[n + 1 for n in notes], options=[o + 1 for o in options],
                         end=v['agent_steps'], done=v['completion'], raw=eps[k[:2] + ({'C': 'A', 'T': 'S'}[system],)]['completion']))
    rows.sort(key=lambda r: (r['first'], r['end'], not r['done']))
    return rows


def r11_timeline():
    data = {s: timeline_rows(s) for s in 'CT'}
    assert len(data['C']) == 31, len(data['C'])
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 112 * MM), gridspec_kw={'wspace': 0.16,
                             'width_ratios': [1, 1]})
    for ax, s, letter in zip(axes, 'CT', 'ab'):
        rows = data[s]
        n = len(rows)
        for i, r in enumerate(rows):
            y = i
            ax.plot([1, r['end']], [y, y], color='#c9c8c1', lw=1.1, zorder=1, solid_capstyle='butt')
            ax.plot([r['first'], r['end']], [y, y], color=INK, lw=1.1, zorder=2, solid_capstyle='butt')
            ax.plot(r['options'], [y] * len(r['options']), '|', markersize=4.5, markeredgewidth=1.3, color=WEBFARM, zorder=3)
            ax.plot(r['first'], y, 'x', markersize=4, markeredgewidth=1.1, color=INK, zorder=4)
            if r['done']:
                ax.plot(r['end'] + 0.35, y, 'o', markersize=4.2, color=WEBFARM, markeredgecolor='white', markeredgewidth=0.4, zorder=4)
            else:
                ax.plot(r['end'] + 0.35, y, 'o', markersize=4.0, markerfacecolor='white', markeredgecolor=INK, markeredgewidth=0.8, zorder=4)
        done = sum(r['done'] for r in rows)
        rescued = sum(r['done'] and not r['raw'] for r in rows)
        firsts = sorted(r['first'] for r in rows)
        med = firsts[n // 2] if n % 2 else (firsts[n // 2 - 1] + firsts[n // 2]) / 2
        ax.set_ylim(n - 0.3, -1.5)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlim(0.5, 16.2)
        ax.set_xticks(range(1, 16))
        ax.set_xlabel('Step of the episode')
        ax.grid(axis='x', zorder=0)
        ax.set_title(f'({letter}) {LABEL[s]}, {n} episodes', loc='left', color=INK)
        ax.text(16.1, -1.1, f'first confident failure:\nmedian step {med:g}\ncompleted {done} of {n}\n'
                f'({rescued} that the agent\nalone failed)', ha='right', va='top', fontsize=7.2, color=INK, linespacing=1.3,
                bbox=dict(facecolor='white', edgecolor=AXIS, linewidth=0.5, pad=2.5))
    handles = [plt.Line2D([], [], color=INK, marker='x', ls='', markersize=4.5, markeredgewidth=1.1),
               plt.Line2D([], [], color=WEBFARM, marker='|', ls='', markersize=6, markeredgewidth=1.4),
               plt.Line2D([], [], color=WEBFARM, marker='o', ls='', markersize=4.6, markeredgecolor='white'),
               plt.Line2D([], [], color=INK, marker='o', ls='', markersize=4.4, markerfacecolor='white'),
               plt.Line2D([], [], color=INK, lw=1.1)]
    fig.legend(handles, ['first confident failure (P ≥ 0.9)', 'recovery option taken', 'task completed', 'task failed',
                         'steps after the first failure'], loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=5,
               handletextpad=0.4, columnspacing=1.3)
    save(fig, 'R11_intervention_timeline')


# ---------------------------------------------------------------- R12 / R13: case studies as annotated storyboards
# Full-width (190 mm) storyboards: few, large, cropped screenshots (headings and links stay readable at print
# size), the story carried by 7.5 pt annotations, the real WebFARM note and options as text, and a numbered
# step list. Screenshot coordinates are pixels of the recorded 1280x720 observations.
import textwrap  # noqa: E402

FIG_W = 190.0


def _mm_axes(fig, height_mm, x, y, w, h):
    """Axes from millimetre coordinates measured from the top-left corner of the figure."""
    return fig.add_axes([x / FIG_W, 1 - (y + h) / height_mm, w / FIG_W, h / height_mm])


def _shot(fig, height_mm, x, y, w, path, crop, marks, border):
    from PIL import Image
    x0, y0, x1, y1 = crop
    img = Image.open(path).convert('RGB').crop(crop)
    h = w * (y1 - y0) / (x1 - x0)
    ax = _mm_axes(fig, height_mm, x, y, w, h)
    ax.imshow(img, extent=(x0, x1, y1, y0), interpolation='lanczos')
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(border); sp.set_linewidth(1.2)
    for m in marks:
        kind = m[0]
        if kind == 'box':
            _, (bx0, by0, bx1, by1), color, dashed = m
            ax.add_patch(matplotlib.patches.FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle='round,pad=0,rounding_size=4',
                                                           fill=False, edgecolor=color, linewidth=1.8,
                                                           linestyle=(0, (3, 1.6)) if dashed else '-', zorder=5))
        elif kind == 'badge':
            _, (bx, by), text, color = m
            ax.text(bx, by, text, ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color, edgecolor='white', linewidth=0.8))
        elif kind == 'label':
            _, (lx, ly), text, color, ha, va = m
            ax.text(lx, ly, text, ha=ha, va=va, fontsize=7.5, color=color, fontweight='bold', zorder=6, linespacing=1.2,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=0.8, alpha=0.95))
    return ax, h


def _band(fig, height_mm, y, text, color):
    ax = _mm_axes(fig, height_mm, 0, y, FIG_W, 6.0)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0.15), 0.012, 0.7, transform=ax.transAxes, color=color))
    ax.text(0.02, 0.5, text, transform=ax.transAxes, ha='left', va='center', fontsize=8.5, color=INK, fontweight='bold')


def _card(fig, height_mm, x, y, w, h, title, lines, color):
    """Small intervention / outcome card (a few short lines, no paragraphs)."""
    ax = _mm_axes(fig, height_mm, x, y, w, h)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(color); sp.set_linewidth(1.4)
    n = len(lines) + 1
    top = (h - n * 4.6) / 2 + 2.3
    ax.text(w / 2, top, title, ha='center', va='center', fontsize=8.5, color=color, fontweight='bold')
    for k, line in enumerate(lines, 1):
        ax.text(w / 2, top + k * 4.6, line, ha='center', va='center', fontsize=7.5, color=INK)


def _arrow(fig, height_mm, start, end):
    (xa, ya), (xb, yb) = start, end
    fig.add_artist(matplotlib.patches.FancyArrowPatch((xa / FIG_W, 1 - ya / height_mm), (xb / FIG_W, 1 - yb / height_mm),
                                                      transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9, color=INK, lw=0.9))


def trajectory_figure(name, rows, cols=2):
    """Rows of 16:9 frames (screenshot or card) with a short label under each, as in WebVoyager / SeeAct figures.

    Two large frames per row (92 mm each) so the web pages stay legible; detail frames are zoomed crops."""
    col_w = 92.0 if cols == 2 else 60.0
    gap = (FIG_W - cols * col_w) / (cols - 1)
    frame_h = col_w * 9 / 16
    row_h = 6.5 + frame_h + 9.5
    height_mm = row_h * len(rows) + 2
    fig = plt.figure(figsize=(FIG_W * MM, height_mm * MM))
    for r, row in enumerate(rows):
        y0 = r * row_h
        _band(fig, height_mm, y0, row['title'], row['color'])
        top = y0 + 6.5
        for c, frame in enumerate(row['frames']):
            x = c * (col_w + gap)
            if frame['kind'] == 'shot':
                crop = frame['crop']
                assert abs((crop[2] - crop[0]) * 9 - (crop[3] - crop[1]) * 16) < 16, crop  # 16:9 frames
                _shot(fig, height_mm, x, top, col_w, frame['path'], crop, frame.get('marks', []), frame.get('border', row['color']))
            else:
                _card(fig, height_mm, x + 4, top + 2, col_w - 8, frame_h - 4, frame['title'], frame['lines'], frame['color'])
            fig.text((x + col_w / 2) / FIG_W, 1 - (top + frame_h + 1.2) / height_mm, frame['caption'], ha='center', va='top',
                     fontsize=7.5, color=INK, linespacing=1.25)
            if c < cols - 1:
                _arrow(fig, height_mm, (x + col_w + 0.6, top + frame_h / 2), (x + col_w + gap - 0.6, top + frame_h / 2))
    save(fig, name)


def _mark_box(box, color, dashed=False):
    return ('box', box, color, dashed)


def _tag(xy, text, color):
    return ('badge', xy, text, color)


# ---------------------------------------------------------------- R12 / R13: case studies as separate step images
# WebVoyager-style panels: one file per step (90 mm, single-column), orange box + callout on the acted element,
# a red tag where P1 detects the failure, a blue tag on steps driven by a WebFARM recovery option, and the
# step label under the screenshot. Files: docs/figures_v2/<case>/step1..step4.{pdf,svg,png} + <case>_all preview.
ACT = '#e8710a'   # action annotation (orange, as in WebVoyager figures)


def _panel(ax, path, crop, marks):
    from PIL import Image
    x0, y0, x1, y1 = crop
    img = Image.open(path).convert('RGB').crop(crop)
    ax.imshow(img, extent=(x0, x1, y1, y0), interpolation='lanczos')
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('#9a9993'); sp.set_linewidth(0.6)
    for m in marks:
        if m[0] == 'element':      # orange box on the element + callout with leader line
            _, (bx0, by0, bx1, by1), text, (lx, ly), ha = m
            ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor=ACT, linewidth=1.6, zorder=5))
            ax.annotate(text, xy=((bx0 + bx1) / 2, by1 if ly > by1 else by0), xytext=(lx, ly), ha=ha, va='center', fontsize=7.5,
                        color=INK, zorder=7, annotation_clip=False,
                        bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor=ACT, linewidth=1.0),
                        arrowprops=dict(arrowstyle='-', color=ACT, lw=1.0, shrinkA=0, shrinkB=0))
        elif m[0] == 'tag':        # status tag in a corner: red = P1 failure, blue = WebFARM option
            _, text, color, corner = m
            xs = {'tl': (x0 + 0.02 * (x1 - x0), 'left'), 'tr': (x1 - 0.02 * (x1 - x0), 'right'),
                  'br': (x1 - 0.02 * (x1 - x0), 'right')}[corner]
            yy, va = (y1 - 0.05 * (y1 - y0), 'bottom') if corner == 'br' else (y0 + 0.06 * (y1 - y0), 'top')
            ax.text(xs[0], yy, text, ha=xs[1], va=va, fontsize=7.5, color='white', fontweight='bold', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))


def step_images(case, steps):
    """Save each step as its own figure and a 2x2 preview of all four."""
    folder = OUT / case
    w = 90.0
    for i, st in enumerate(steps, 1):
        x0, y0, x1, y1 = st['crop']
        h_img = w * (y1 - y0) / (x1 - x0)
        h = h_img + 9.0
        fig = plt.figure(figsize=(w * MM, h * MM))
        ax = fig.add_axes([0, 9.0 / h, 1, h_img / h])
        _panel(ax, st['path'], st['crop'], st['marks'])
        fig.text(0.5, 7.6 / h, st['caption'], ha='center', va='top', fontsize=7.5, color=INK, linespacing=1.2)
        folder.mkdir(parents=True, exist_ok=True)
        for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
            fig.savefig(folder / f'step{i}.{ext}', pad_inches=0.0, **kw)
        plt.close(fig)
    # preview: 2 x 2 grid at full width, same drawing code
    col, gap, cap = 92.0, FIG_W - 2 * 92.0, 9.0
    hs = [col * (s['crop'][3] - s['crop'][1]) / (s['crop'][2] - s['crop'][0]) for s in steps]
    row_h = [max(hs[0], hs[1]) + cap, max(hs[2], hs[3]) + cap]
    H = sum(row_h) + 2
    fig = plt.figure(figsize=(FIG_W * MM, H * MM))
    for i, st in enumerate(steps):
        r, c = divmod(i, 2)
        top = sum(row_h[:r])
        x = c * (col + gap)
        ax = fig.add_axes([x / FIG_W, 1 - (top + hs[i]) / H, col / FIG_W, hs[i] / H])
        _panel(ax, st['path'], st['crop'], st['marks'])
        fig.text((x + col / 2) / FIG_W, 1 - (top + hs[i] + 1.2) / H, st['caption'], ha='center', va='top', fontsize=7.5, color=INK,
                 linespacing=1.2)
    save(fig, f'{case}_all')
    print('  + step1..step4 in', folder)


def r12_case_browser_use():
    eps = load4()
    A, C = eps[(0, 'wikipedia-wiki-marie-curie', 'A')], eps[(0, 'wikipedia-wiki-marie-curie', 'C')]
    assert not A['completion'] and C['completion'] and C['split'] == 'held_out'
    acts = action_events(C['folder'])
    assert [a.get('action_type') for a in acts[:4]] == ['TYPE', 'CLICK', 'TYPE', 'PRESS_KEY'] and acts[0]['value'] == 'Marie Curie'
    pf = {k: j['assessment'].get('outcome_probabilities', {}).get('FAILURE') for k, j in assessments(C).items()}
    assert abs(pf[1] - 0.9998) < 0.001 and pf[0] < 0.2
    c3 = json.loads((C['folder'] / 'actor-0003-choice.json').read_text())
    assert c3['chosen'] == 4 and c3['picked']['label'].startswith("Search this site for 'Marie Curie'")
    c4 = json.loads((C['folder'] / 'actor-0004-choice.json').read_text())
    assert c4['chosen'] == 1 and c4['picked']['label'] == 'Press Enter to submit the search'
    obs = lambda k: json.loads((C['folder'] / f'observation-{k:04d}.json').read_text())['url']
    assert obs(2).endswith('Main_Page') and obs(4).endswith('/wiki/Marie_Curie')
    img = RUNS['C'] / 'browser/20261004T112703/wikipedia-wiki-marie-curie-r0-C/images'
    crop = (0, 0, 960, 540)
    box = (262, 16, 670, 50)
    steps = [dict(path=img / '0001.png', crop=crop, caption='Step 1: Type [Search Wikipedia]: Marie Curie',
                  marks=[('element', box, '[Search Wikipedia]', (520, 150), 'center')]),
             dict(path=img / '0002.png', crop=crop, caption='Step 2: Click [span]: page unchanged,\nP1 detects the failure',
                  marks=[('tag', f'P1: failure ({pf[1]:.2f})', HURT, 'br')]),
             dict(path=img / '0003.png', crop=crop, caption='Step 3: Type [Search Wikipedia]: Marie Curie\n(WebFARM option "search the site")',
                  marks=[('element', (240, 17, 668, 49), '[Search Wikipedia]', (820, 140), 'center'),
                         ('tag', 'WebFARM option', WEBFARM, 'br')]),
             dict(path=img / '0004.png', crop=crop, caption='Step 4: Press Enter (WebFARM option):\ntarget article reached',
                  marks=[('element', (258, 90, 410, 128), '[Marie Curie]', (520, 109), 'left'),
                         ('tag', 'WebFARM option', WEBFARM, 'br')])]
    step_images('R12_case_browser_use_wikipedia', steps)


def r13_case_seeact():
    eps = load4()
    S, T = eps[(0, 'epa-climate-change', 'S')], eps[(0, 'epa-climate-change', 'T')]
    assert not S['completion'] and T['completion']
    t_acts = action_events(T['folder'])
    assert [a['seeact_action'] for a in t_acts[:5]] == ['TYPE', 'CLICK', 'TYPE', 'PRESS ENTER', 'GOTO']
    assert t_acts[4]['url_after'].rstrip('/').endswith('epa.gov/climate-change')
    pf = {k: j['assessment'].get('outcome_probabilities', {}).get('FAILURE') for k, j in assessments(T).items()}
    assert abs(pf[1] - 0.988) < 0.002 and pf[0] < 0.01
    assert json.loads((T['folder'] / 'actor-0003-choice.json').read_text())['chosen'] == 1
    assert json.loads((T['folder'] / 'actor-0005-choice.json').read_text())['picked']['label'].startswith("Open the link 'Climate Change'")
    img = RUNS['T'] / 'browser/20261014T105632/epa-climate-change-r0-T/images'
    right, left = (0, 0, 1280, 720), (0, 0, 960, 540)
    box = (982, 56, 1222, 96)
    steps = [dict(path=img / '0001.png', crop=right, caption='Step 1: Type [search box]: climate change',
                  marks=[('element', box, '[search box]', (960, 210), 'center')]),
             dict(path=img / '0002.png', crop=right, caption='Step 2: Click [search box] again: page unchanged,\nP1 detects the failure',
                  marks=[('element', box, '[search box]', (960, 210), 'center'), ('tag', f'P1: failure ({pf[1]:.2f})', HURT, 'br')]),
             dict(path=img / '0003.png', crop=right, caption='Steps 3–4: Type [search box], Press Enter\n(WebFARM options "search the site", "Enter")',
                  marks=[('element', box, '[search box]', (960, 210), 'center'), ('tag', 'WebFARM option', WEBFARM, 'br')]),
             dict(path=img / '0005.png', crop=left, caption="Step 5: Go to [Climate Change] (WebFARM option):\ntarget page reached",
                  marks=[('element', (22, 218, 352, 264), '[Climate Change]', (560, 241), 'left'), ('tag', 'WebFARM option', WEBFARM, 'br')])]
    step_images('R13_case_seeact_epa', steps)


# ---------------------------------------------------------------- A1: per-task matrix, four systems
def a1_per_task():
    eps = load4()
    tasks = list(dict.fromkeys(t for _, t in keys()))
    split = {t: eps[(0, t, 'A')]['split'] for t in tasks}
    groups = [('Development tasks', sorted([t for t in tasks if split[t] == 'development'], key=lambda t: v1.TASK_LABELS[t].lower())),
              ('Held-out tasks', sorted([t for t in tasks if split[t] == 'held_out'], key=lambda t: v1.TASK_LABELS[t].lower()))]
    cols = [(s, r) for s in 'ACST' for r in (0, 1)]
    xs = {}
    x = 0.0
    for i, (s, r) in enumerate(cols):
        if i and i % 2 == 0:
            x += 0.45
        xs[(s, r)] = x
        x += 1
    fig = plt.figure(figsize=(190 * MM, 200 * MM))
    ax = fig.add_axes([0.3, 0.03, 0.62, 0.9])
    y = 0.0
    for gname, ts in groups:
        ax.text(-0.3, y, gname, ha='right', va='center', fontsize=8, color=INK, fontweight='bold')
        y += 1
        for t in ts:
            for (s, r), xx in xs.items():
                ok = eps[(r, t, s)]['completion']
                ax.add_patch(plt.Rectangle((xx + 0.08, y - 0.4), 0.84, 0.8, facecolor=COLOR[s] if ok else 'white',
                                           edgecolor=COLOR[s] if ok else '#b8b7b0', linewidth=0.6))
            ax.text(-0.3, y, v1.TASK_LABELS[t], ha='right', va='center', fontsize=7, color=INK)
            for (agent, raw, fw), xx in zip(AGENTS, (x + 0.3, x + 1.9)):
                net = sum(eps[(r, t, fw)]['completion'] - eps[(r, t, raw)]['completion'] for r in (0, 1))
                if net:
                    ax.text(xx + 0.4, y, f'{net:+d}'.replace('-', '−'), ha='center', va='center', fontsize=7, color=INK,
                            fontweight='bold')
            y += 1
        y += 0.6
    for s in 'ACST':
        xm = (xs[(s, 0)] + xs[(s, 1)] + 1) / 2
        ax.text(xm, -1.7, LABEL[s].replace(' + ', '\n+ '), ha='center', va='bottom', fontsize=7.5, color=INK, fontweight='bold',
                linespacing=1.15)
        for r in (0, 1):
            ax.text(xs[(s, r)] + 0.5, -0.75, f'run {r + 1}', ha='center', va='center', fontsize=7, color=INK)
    for (agent, _, _), xx in zip(AGENTS, (x + 0.3, x + 1.9)):
        ax.text(xx + 0.4, -1.7, 'net\n' + agent.replace(' ', '\n'), ha='center', va='bottom', fontsize=7, color=INK, linespacing=1.15)
    ax.set_xlim(-0.2, x + 2.7)
    ax.set_ylim(y, -2.6)
    ax.axis('off')
    fig.legend(legend_patches([(BASE, None), (WEBFARM, None), ('white', None)]),
               ['completed (agent alone)', 'completed (agent + WebFARM)', 'failed'], loc='lower center',
               bbox_to_anchor=(0.6, -0.005), ncol=3, handlelength=1.2, columnspacing=1.6)
    fig.legends[0].legend_handles[2].set_edgecolor('#b8b7b0')
    fig.legends[0].legend_handles[2].set_linewidth(0.6)
    save(fig, 'A1_per_task_four_systems')


# ---------------------------------------------------------------- A2: per-site, four systems
def a2_per_site():
    eps = load4()
    suite = {t['id']: t for t in json.loads((ROOT / 'configs/eval/task2/web_tasks_v2.json').read_text())['tasks']}
    site = {}
    for (r, t, s), v in eps.items():
        d = v1.SITE_NAMES[suite[t]['dataset_domain']]
        site.setdefault(d, {x: [] for x in 'ACST'} | {'tasks': set()})
        site[d][s].append(v['completion'])
        site[d]['tasks'].add(t)
    rate = {d: {s: 100 * sum(v[s]) / len(v[s]) for s in 'ACST'} for d, v in site.items()}
    order = sorted(site, key=lambda d: (-(rate[d]['C'] - rate[d]['A'] + rate[d]['T'] - rate[d]['S']), d))
    fig, ax = plt.subplots(figsize=(140 * MM, 165 * MM))
    fig.subplots_adjust(left=0.26, right=0.95)
    y, yt = 0.0, []
    for d in order:
        for agent, raw, fw in AGENTS:
            a, c = rate[d][raw], rate[d][fw]
            ax.plot([a, c], [y, y], color=INK if a != c else AXIS, lw=0.9 if a != c else 0.6, ls=DASH[raw], zorder=2)
            if a != c:
                ax.plot(a, y, MARK[raw], markersize=5, color=BASE, markeredgecolor='white', markeredgewidth=0.6, zorder=3)
                ax.plot(c, y, MARK[fw], markersize=5, color=WEBFARM, markeredgecolor='white', markeredgewidth=0.6, zorder=4)
                ax.text(max(a, c) + 3, y, f'{c - a:+.0f}'.replace('-', '−'), ha='left', va='center', fontsize=7, color=INK,
                        fontweight='bold')
            else:
                ax.plot(a, y, MARK[raw], markersize=5.8, fillstyle='left', markerfacecolor=BASE, markerfacecoloralt=WEBFARM,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=4)
            y += 0.42
        yt.append(y - 0.63)
        y += 0.75
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{d} ({len(site[d]["tasks"])} task{"s" if len(site[d]["tasks"]) > 1 else ""})' for d in order], fontsize=7.2)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(y - 0.5, -0.6)
    ax.set_xlim(-4, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.spines['bottom'].set_bounds(0, 100)
    ax.set_xlabel('Task completion (%), both repeats')
    ax.grid(axis='x', zorder=0)
    handles = [plt.Line2D([], [], color=INK, lw=0.9, ls=DASH['A'], marker='o', markersize=5, markerfacecolor=BASE, markeredgecolor='white'),
               plt.Line2D([], [], color=INK, lw=0.9, ls=DASH['S'], marker='s', markersize=5, markerfacecolor=BASE, markeredgecolor='white'),
               plt.Line2D([], [], marker='o', ls='', markersize=5, color=BASE), plt.Line2D([], [], marker='o', ls='', markersize=5, color=WEBFARM)]
    ax.legend(handles, ['Browser Use (upper row)', 'SeeAct (lower row)', 'agent alone', 'agent + WebFARM'], loc='upper center',
              bbox_to_anchor=(0.42, 1.06), ncol=2, handlelength=2.4, columnspacing=1.6)
    assert sum(len(v['A']) for v in site.values()) == 84
    save(fig, 'A2_per_site_four_systems')


# ---------------------------------------------------------------- A3: recovery funnel, both agents
def a3_funnel():
    eps = load4()
    stages = ['All episodes', 'Confident failure detected\n(P ≥ 0.9, recovery opened)', 'Recovery options offered',
              'Actor took an option', 'Recovery verified by P1', 'Task completed']
    counts = {}
    for s in 'CT':
        rows = [recovery_trace(v, s) for k, v in eps.items() if k[2] == s]
        counts[s] = [len(rows), sum(x['opened'] for x in rows), sum(x['opened'] and x['offered'] for x in rows),
                     sum(x['opened'] and x['taken'] for x in rows), sum(x['opened'] and x['verified'] for x in rows),
                     sum(x['opened'] and x['completed'] for x in rows)]
    assert counts['C'] == [84, 31, 31, 31, 30, 26], counts['C']
    fig, ax = plt.subplots(figsize=(140 * MM, 74 * MM))
    fig.subplots_adjust(left=0.3)
    h = 0.36
    for i, label in enumerate(stages):
        y = i
        for off, s, hatch in ((-h / 2 - 0.02, 'C', None), (h / 2 + 0.02, 'T', '////')):
            n = counts[s][i]
            ax.barh(y + off, n, height=h, color=WEBFARM if i else WEBFARM_LIGHT, hatch=hatch, edgecolor='white', linewidth=0, zorder=2)
            ax.text(n + 1, y + off, str(n), ha='left', va='center', fontsize=7.5, color=INK, fontweight='bold')
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=7.5)
    ax.set_ylim(len(stages) - 0.4, -0.6)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 92)
    ax.set_xticks(range(0, 81, 20))
    ax.spines['bottom'].set_bounds(0, 80)
    ax.set_xlabel('Episodes (84 per agent)')
    ax.grid(axis='x', zorder=0)
    ax.legend(legend_patches([(WEBFARM, None), (WEBFARM, '////')]), ['WebFARM on Browser Use', 'WebFARM on SeeAct'],
              loc='upper center', bbox_to_anchor=(0.42, -0.2), ncol=2, handlelength=1.6, columnspacing=2.0)
    save(fig, 'A3_recovery_funnel_both_agents')


FIGURES = {'R01': r01_task1, 'R02': r02_confusion, 'R03': r03_main, 'R04': r04_slope, 'R05': r05_paired, 'R06': r06_cost,
           'R07': r07_step_budget, 'R08': r08_options, 'R09': r09_failure_modes, 'R10': r10_detector, 'R11': r11_timeline,
           'R12': r12_case_browser_use, 'R13': r13_case_seeact, 'A1': a1_per_task, 'A2': a2_per_site, 'A3': a3_funnel}

if __name__ == '__main__':
    for name in sys.argv[1:] or list(FIGURES):
        FIGURES[name]()
