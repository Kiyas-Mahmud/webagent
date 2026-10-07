"""Results-section figures for the thesis, computed from the raw run files.

Usage:  .venv/bin/python scripts/make_results_figures.py [fig1 fig2 ...]   (default: all)
Output: docs/figures/<name>.pdf (vector, for the paper) and .png (preview).

Every number is recomputed from the saved episodes / CSVs and checked against
the values reported in docs/TASK2_WEB_V3_RESULTS.md, so figures and tables
cannot drift apart. Sizes follow Elsevier artwork rules: final width 90 / 140 /
190 mm, text 7-9 pt at final size, Liberation Sans (Arial-metric).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.task2-assets/comparison/web-v3'
OUT = ROOT / 'docs/figures'
MM = 1 / 25.4

# ---- palette (validated: blue/grey pair; ink and chrome from the reference palette) ----
OURS = '#2a78d6'       # Browser Use + Ours
OURS_LIGHT = '#9ec5f4' # same hue, lighter step: totals of Ours episodes
BASE = '#8c8b85'       # Browser Use alone
HURT = '#e34948'
SAME = '#d9d8d2'
INK = '#111111'
# All text is near-black for legibility in print and on projectors (user rule: no light text).
INK2 = INK
MUTED = INK
GRID = '#e8e7e1'
AXIS = '#c3c2b7'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'figure.dpi': 150,
})


# ---------------------------------------------------------------- data
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def sign_p(helped, hurt):
    n = helped + hurt
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(helped, hurt) + 1)) / 2 ** n) if n else 1.0


def load_episodes():
    plan = json.loads((RUN / 'plan.json').read_text())
    eps = {}
    for e in plan['episodes']:
        p = RUN / 'episodes' / f"{e['repeat']}-{e['task']}-{e['system']}" / 'result.json'
        r = json.loads(p.read_text())
        eps[(e['repeat'], e['task'], e['system'])] = dict(r, split=e['split'], folder=p.parent)
    assert len(eps) == 168, f'expected 168 finished episodes, found {len(eps)}'
    return plan, eps


def pairs_by_split(eps):
    out = {'Development': [], 'Held-out': []}
    for (r, t, s), v in eps.items():
        if s == 'A':
            key = 'Development' if v['split'] == 'development' else 'Held-out'
            out[key].append((v['completion'], eps[(r, t, 'C')]['completion']))
    out['All'] = out['Development'] + out['Held-out']
    return out


def check(name, got, want, tol=0.05):
    if abs(got - want) > tol:
        raise AssertionError(f'{name}: computed {got}, documented {want}')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


# ---------------------------------------------------------------- figures
def fig1_completion():
    """Main result: completion rate with Wilson 95% CI, per task set."""
    _, eps = load_episodes()
    groups = pairs_by_split(eps)
    documented = {'Development': (70.0, 87.5, 0.039), 'Held-out': (75.0, 84.1, 0.219), 'All': (72.6, 85.7, 0.007)}

    fig, ax = plt.subplots(figsize=(140 * MM, 68 * MM))
    width, gap = 0.34, 0.02
    for i, (name, pairs) in enumerate(groups.items()):
        n = len(pairs)
        a = sum(x for x, _ in pairs)
        c = sum(y for _, y in pairs)
        helped = sum((not x) and y for x, y in pairs)
        hurt = sum(x and (not y) for x, y in pairs)
        pa, la, ua = wilson(a, n)
        pc, lc, uc = wilson(c, n)
        p = sign_p(helped, hurt)
        check(f'{name} baseline', pa, documented[name][0])
        check(f'{name} ours', pc, documented[name][1])
        check(f'{name} p', p, documented[name][2], tol=0.0005)
        for x, val, lo, hi, color in ((i - width / 2 - gap / 2, pa, la, ua, BASE), (i + width / 2 + gap / 2, pc, lc, uc, OURS)):
            ax.bar(x, val, width, color=color, zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=0.7, zorder=3)
            ax.plot([x - 0.05, x + 0.05], [lo, lo], color=INK, lw=0.7, zorder=3)
            ax.plot([x - 0.05, x + 0.05], [hi, hi], color=INK, lw=0.7, zorder=3)
            ax.text(x, 4, f'{val:.1f}', ha='center', va='bottom', fontsize=7.5,
                    color='white', fontweight='bold', zorder=4)
        # difference bracket above the pair
        top = max(ua, uc) + 4
        x0, x1 = i - width / 2 - gap / 2, i + width / 2 + gap / 2
        ax.plot([x0, x0, x1, x1], [top - 1.5, top, top, top - 1.5], color=INK2, lw=0.6)
        ptxt = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
        ax.text(i, top + 1.2, f'+{pc - pa:.1f} pts, {ptxt}', ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f'{k}\n({len(v)} pairs)' for k, v in groups.items()])
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_ylabel('Task completion (%)')
    ax.grid(axis='y', zorder=0)
    ax.set_xlim(-0.6, 2.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=OURS)]
    ax.legend(handles, ['Browser Use (Qwen2.5-VL-7B)', 'Browser Use + Ours'], loc='upper center',
              bbox_to_anchor=(0.5, 1.13), ncol=2, handlelength=1.2, columnspacing=1.6)
    save(fig, 'fig1_task_completion')


def fig2_paired_outcomes():
    """Discordant pairs and rescue/harm rates per task set."""
    _, eps = load_episodes()
    groups = pairs_by_split(eps)
    documented = {'Development': (8, 1, 31), 'Held-out': (5, 1, 38), 'All': (13, 2, 69)}
    fig, ax = plt.subplots(figsize=(140 * MM, 46 * MM))
    names = list(groups)[::-1]          # All at the bottom reads as the total
    for row, name in enumerate(names):
        pairs = groups[name]
        helped = sum((not a) and c for a, c in pairs)
        hurt = sum(a and (not c) for a, c in pairs)
        same = len(pairs) - helped - hurt
        assert (helped, hurt, same) == documented[name], (name, helped, hurt, same)
        a_fail = sum(not a for a, _ in pairs)
        a_ok = len(pairs) - a_fail
        left = 0
        for count, color, label, text_color in ((helped, OURS, 'Helped', 'white'), (same, SAME, 'Same', INK2),
                                                (hurt, HURT, 'Hurt', 'white')):
            ax.barh(row, count, left=left, height=0.56, color=color, edgecolor='white', linewidth=1.0, zorder=2)
            if count >= 3:
                ax.text(left + count / 2, row, str(count), ha='center', va='center', fontsize=7.5,
                        color=text_color, fontweight='bold' if text_color == 'white' else 'normal')
            left += count
        # the hurt segment is too thin to hold its count: print it just past the bar end
        ax.text(left + 0.8, row, str(hurt), ha='left', va='center', fontsize=7.5, color=INK, fontweight='bold')
        ax.text(left + 4.5, row, f'rescue {100 * helped / a_fail:.0f}% ({helped}/{a_fail})  ·  '
                                 f'harm {100 * hurt / a_ok:.0f}% ({hurt}/{a_ok})',
                ha='left', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_xlim(0, 125)
    ax.set_xticks(range(0, 91, 10))
    ax.spines['bottom'].set_bounds(0, 90)
    ax.set_xlabel('Number of task pairs (Browser Use vs. Browser Use + Ours)', x=0.36)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-0.6, len(names) - 0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, SAME, HURT)]
    ax.legend(handles, ['Helped: only Ours completed', 'Same outcome', 'Hurt: only Browser Use completed'],
              loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=3, handlelength=1.2, columnspacing=1.4)
    save(fig, 'fig2_paired_outcomes')


TASK_LABELS = {
    'wikipedia-wiki-alan-turing': 'Wikipedia: Alan Turing', 'wikipedia-wiki-eiffel-tower': 'Wikipedia: Eiffel Tower',
    'wikipedia-wiki-photosynthesis': 'Wikipedia: photosynthesis', 'wiktionary-wiki-serendipity': 'Wiktionary: serendipity',
    'wiktionary-wiki-ephemeral': 'Wiktionary: ephemeral', 'arxiv-list-cs-ai': 'arXiv: cs.AI list',
    'debian-security': 'Debian: security', 'creativecommons-cc-licenses': 'Creative Commons: licenses',
    'gov-renew-adult-passport': 'GOV.UK: renew passport', 'gov-bank-holidays': 'GOV.UK: bank holidays',
    'ted-talks': 'TED: talks', 'jquery-download': 'jQuery: download', 'jquery-api-jquery-com': 'jQuery: API docs',
    'epa-climate-change': 'EPA: climate change', 'ietf-process-rfcs': 'IETF: RFCs',
    'stanford-admission-stanford-edu': 'Stanford: admission', 'yale-admissions': 'Yale: admissions',
    'europa-principles-countries-history-eu-countrie': 'Europa: EU countries', 'arxiv-list-math-recent': 'arXiv: math list',
    'debian-news': 'Debian: news', 'coursera-degrees': 'Coursera: degrees', 'cpanel-pricing': 'cPanel: pricing',
    'epa-laws-regulations': 'EPA: laws & regulations', 'epa-recycle': 'EPA: recycling', 'europa-institutions': 'Europa: institutions',
    'github-pricing': 'GitHub: pricing', 'gitlab-pricing': 'GitLab: pricing', 'gov-uk-apply-driving-licence': 'GOV.UK: driving licence',
    'gov-uk-state-pension-age': 'GOV.UK: state pension age', 'ietf-about': 'IETF: about', 'ietf-meetings-upcoming': 'IETF: meetings',
    'nextcloud-install': 'Nextcloud: install', 'noaa-weather': 'NOAA: weather', 'stripe-pricing': 'Stripe: pricing',
    'wikipedia-random': 'Wikipedia: community portal', 'wikipedia-wiki-black-hole': 'Wikipedia: black hole',
    'wikipedia-wiki-marie-curie': 'Wikipedia: Marie Curie', 'wiktionary-wiki-ubiquitous': 'Wiktionary: ubiquitous',
    'wisc-admissions': 'UW-Madison: admissions', 'wordpress-download': 'WordPress: download',
    'wordpress-plugins': 'WordPress: plugins', 'yale-about': 'Yale: about',
}


def fig3_per_task():
    """Per-task outcome matrix: 42 tasks x (Browser Use r1, r2, Ours r1, r2)."""
    plan, eps = load_episodes()
    tasks = list(dict.fromkeys(e['task'] for e in plan['episodes']))
    assert set(tasks) == set(TASK_LABELS), set(tasks) ^ set(TASK_LABELS)
    split = {e['task']: e['split'] for e in plan['episodes']}

    def row(t):
        a = [eps[(r, t, 'A')]['completion'] for r in (0, 1)]
        c = [eps[(r, t, 'C')]['completion'] for r in (0, 1)]
        net = sum((not x) and y for x, y in zip(a, c)) - sum(x and (not y) for x, y in zip(a, c))
        group = 0 if net > 0 else 1 if net < 0 else 2 if not any(a + c) else 3
        return a, c, net, group

    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 100 * MM), gridspec_kw={'wspace': 0.95})
    fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.09)
    totals = {'helped': 0, 'hurt': 0}
    for ax, (name, key) in zip(axes, (('Development tasks (n = 20)', 'development'), ('Held-out tasks (n = 22)', 'held_out'))):
        ts = [t for t in tasks if split[t] == key]
        ts.sort(key=lambda t: (row(t)[3], -row(t)[2], TASK_LABELS[t]))
        n = len(ts)
        for i, t in enumerate(ts):
            a, c, net, _ = row(t)
            y = n - 1 - i
            for j, (done, color) in enumerate(zip(a + c, (BASE, BASE, OURS, OURS))):
                x = j + (0.35 if j >= 2 else 0)
                if done:
                    ax.add_patch(plt.Rectangle((x + 0.06, y + 0.08), 0.88, 0.84, color=color, lw=0))
                else:
                    ax.add_patch(plt.Rectangle((x + 0.06, y + 0.08), 0.88, 0.84, facecolor='white',
                                               edgecolor=AXIS, lw=0.6))
            if net:
                ax.text(4.6, y + 0.5, f'{net:+d}'.replace('-', '\u2212'), ha='left', va='center', fontsize=7.5,
                        color=INK, fontweight='bold')
                totals['helped' if net > 0 else 'hurt'] += abs(net)
            ax.text(-0.25, y + 0.5, TASK_LABELS[t], ha='right', va='center', fontsize=7, color=INK)
        ax.set_xlim(-0.1, 5.2)
        ax.set_ylim(-0.1, n + 1.75)
        for x, label in ((1.0, 'Browser Use'), (3.35, 'Browser Use + Ours')):
            ax.text(x, n + 1.3, label, ha='center', va='center', fontsize=7.5, color=INK)
        for j in range(4):
            x = j + (0.35 if j >= 2 else 0)
            ax.text(x + 0.5, n + 0.3, f'run {j % 2 + 1}', ha='center', va='center', fontsize=6.5, color=MUTED)
        ax.text(4.75, n + 0.3, 'net', ha='center', va='center', fontsize=6.5, color=MUTED)
        ax.set_title(name, loc='left', fontsize=8.5, color=INK, pad=2, x=-0.9)
        ax.axis('off')
    assert totals == {'helped': 13, 'hurt': 2}, totals
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=OURS),
               plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=AXIS, lw=0.6)]
    fig.legend(handles, ['Completed (Browser Use)', 'Completed (Browser Use + Ours)', 'Failed'],
               loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=3, handlelength=1.2, columnspacing=1.8)
    fig.text(0.5, 0.0, 'net = pairs where only Ours completed minus pairs where only Browser Use completed',
             ha='center', fontsize=6.5, color=MUTED)
    save(fig, 'fig3_per_task_outcomes')


def fig4_cost():
    """(a) wall-clock time per episode; (b) steps on pairs both systems completed."""
    import statistics as st
    _, eps = load_episodes()
    times = {s: [v['elapsed_seconds'] for k, v in eps.items() if k[2] == s] for s in 'AC'}
    check('mean time A', st.mean(times['A']), 220, tol=0.6)
    check('mean time C', st.mean(times['C']), 175, tol=0.6)
    both = [(r, t) for (r, t, s) in eps if s == 'A' and eps[(r, t, 'A')]['completion'] and eps[(r, t, 'C')]['completion']]
    steps = {s: [eps[p + (s,)]['agent_steps'] for p in both] for s in 'AC'}
    assert len(both) == 59
    check('steps A', st.mean(steps['A']), 3.08, tol=0.006)
    check('steps C', st.mean(steps['C']), 3.32, tol=0.006)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(140 * MM, 56 * MM), gridspec_kw={'width_ratios': [1, 1.25], 'wspace': 0.42})
    # (a) time distribution
    for i, (s, color) in enumerate((('A', BASE), ('C', OURS))):
        box = ax1.boxplot(times[s], positions=[i], widths=0.42, orientation='vertical', patch_artist=True, showfliers=True,
                          medianprops=dict(color='white', lw=1.2), whiskerprops=dict(color=color, lw=0.8),
                          capprops=dict(color=color, lw=0.8),
                          flierprops=dict(marker='o', markersize=2.5, markerfacecolor=color, markeredgecolor='white',
                                          markeredgewidth=0.3))
        box['boxes'][0].set(facecolor=color, edgecolor=color)
        m = st.mean(times[s])
        ax1.plot(i, m, marker='D', markersize=3.6, color='white', markeredgecolor=INK, markeredgewidth=0.6, zorder=4)
        ax1.text(i - 0.26, m, f'{m:.0f} s', ha='right', va='center', fontsize=7, color=INK)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Browser Use', 'Browser Use\n+ Ours'])
    ax1.tick_params(axis='x', length=0)
    ax1.set_xlim(-0.65, 1.45)
    ax1.set_ylabel('Wall-clock time per episode (s)')
    ax1.grid(axis='y', zorder=0)
    ax1.set_title('(a) All 84 episodes per system', loc='left', color=INK)
    ax1.plot([], [], marker='D', markersize=3.6, color='white', markeredgecolor=INK, markeredgewidth=0.6, ls='',
             label='mean')
    ax1.legend(loc='upper left', handlelength=1, borderaxespad=0.2)
    # (b) step distribution on both-completed pairs
    values = range(1, max(steps['A'] + steps['C']) + 1)
    w = 0.38
    for off, s, color, label in ((-w / 2 - 0.02, 'A', BASE, 'Browser Use'), (w / 2 + 0.02, 'C', OURS, 'Browser Use + Ours')):
        counts = [steps[s].count(v) for v in values]
        ax2.bar([v + off for v in values], counts, w, color=color, label=f'{label} (mean {st.mean(steps[s]):.2f})', zorder=2)
    ax2.set_xticks(list(values))
    ax2.set_xlabel('Steps to complete the task')
    ax2.set_ylabel('Number of episodes')
    ax2.grid(axis='y', zorder=0)
    ax2.set_title(f'(b) Pairs both systems completed (n = {len(both)})', loc='left', color=INK)
    ax2.set_ylim(0, 31)
    ax2.set_yticks(range(0, 26, 5))
    ax2.legend(loc='upper right', handlelength=1.1, borderaxespad=0.2)
    save(fig, 'fig4_cost')


def recovery_trace(folder):
    """Per-episode recovery facts for system C, read from the saved files."""
    ass = [json.loads(a.read_text())['assessment'] for a in sorted(folder.glob('assessment-*.json'))]
    choices = [json.loads(x.read_text()) for x in folder.glob('actor-*-choice.json')]
    return {'opened': bool(list(folder.glob('note-*.json'))),        # confident failure on a non-terminal step
            'offered': bool(choices), 'taken': any(c['chosen'] for c in choices),
            'verified': any(a['phase'] == 'recovery_assessment' and a['signals']['outcome_label'] == 'SUCCESS' for a in ass)}


def fig5_recovery():
    """(a) Episode-level recovery funnel; (b) where the gain comes from."""
    _, eps = load_episodes()
    rows = []
    for (r, t, s), v in eps.items():
        if s == 'C':
            rows.append(dict(recovery_trace(v['folder']), completed=v['completion'], baseline=eps[(r, t, 'A')]['completion']))
    stages = [('All episodes', len(rows)),
              ('Confident failure detected\n(P ≥ 0.9, recovery opened)', sum(x['opened'] for x in rows)),
              ('Recovery options offered', sum(x['offered'] for x in rows)),
              ('Actor took an option', sum(x['taken'] for x in rows)),
              ('Recovery verified by P1', sum(x['opened'] and x['verified'] for x in rows)),
              ('Task completed', sum(x['opened'] and x['completed'] for x in rows))]
    assert [n for _, n in stages] == [84, 31, 31, 31, 30, 26], stages
    opened = [x for x in rows if x['opened']]
    silent = [x for x in rows if not x['opened']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(190 * MM, 62 * MM), gridspec_kw={'width_ratios': [1.25, 1], 'wspace': 0.55})
    # (a) funnel
    ys = list(range(len(stages)))[::-1]
    for y, (label, n) in zip(ys, stages):
        ax1.barh(y, n, height=0.62, color=OURS if y < ys[0] else OURS_LIGHT, zorder=2)
        ax1.text(n + 1.2, y, f'{n}', ha='left', va='center', fontsize=7.5, color=INK, fontweight='bold')
    ax1.set_yticks(ys)
    ax1.set_yticklabels([label for label, _ in stages], fontsize=7.5)
    ax1.tick_params(axis='y', length=0)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlim(0, 92)
    ax1.set_xticks(range(0, 85, 20))
    ax1.spines['bottom'].set_bounds(0, 80)
    ax1.set_xlabel('Browser Use + Ours episodes')
    ax1.grid(axis='x', zorder=0)
    ax1.set_title('(a) Recovery pipeline (84 episodes)', loc='left', color=INK, x=-0.62)
    # (b) completion where the framework acted vs stayed silent
    groups = (('Recovery opened', opened), ('Detector silent', silent))
    w = 0.34
    for i, (name, g) in enumerate(groups):
        n = len(g)
        a = 100 * sum(x['baseline'] for x in g) / n
        c = 100 * sum(x['completed'] for x in g) / n
        helped = sum((not x['baseline']) and x['completed'] for x in g)
        hurt = sum(x['baseline'] and not x['completed'] for x in g)
        for off, val, color in ((-w / 2 - 0.02, a, BASE), (w / 2 + 0.02, c, OURS)):
            ax2.bar(i + off, val, w, color=color, zorder=2)
            ax2.text(i + off, val + 1.5, f'{val:.0f}', ha='center', va='bottom', fontsize=7.5, color=INK)
        ax2.text(i, 104, f'helped {helped}\nhurt {hurt}', ha='center', va='bottom', fontsize=7.5, color=INK,
                 linespacing=1.25)
    assert (sum((not x['baseline']) and x['completed'] for x in opened), sum(x['baseline'] and not x['completed'] for x in opened)) == (12, 0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f'{name}\n({len(g)} pairs)' for name, g in groups])
    ax2.tick_params(axis='x', length=0)
    ax2.set_ylim(0, 122)
    ax2.set_yticks(range(0, 101, 20))
    ax2.spines['left'].set_bounds(0, 100)
    ax2.set_ylabel('Task completion (%)')
    ax2.grid(axis='y', zorder=0)
    ax2.set_xlim(-0.6, 1.6)
    ax2.set_title('(b) Completion by whether the framework acted', loc='left', color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=OURS)]
    fig.legend(handles, ['Browser Use', 'Browser Use + Ours'], loc='lower center', bbox_to_anchor=(0.72, -0.1),
               ncol=2, handlelength=1.2, columnspacing=1.6)
    save(fig, 'fig5_recovery_pipeline')


def task1_rows(phase):
    """Task 1 pilot rows (240 interaction / 120 recovery cases): prompted base models and trained heads."""
    def read(path):
        return list(csv.DictReader(open(ROOT / path, encoding='utf-8-sig')))
    out = []
    for backbone, path in (('Qwen2.5-VL-7B', 'results/task1_qwen25_dual_v1'), ('InternVL3.5-8B', 'results/task1_internvl_dual_v2')):
        for r in read(f'{path}/{phase}_metrics.csv'):
            if r['weight_variant'] == 'base' or r['method'] == 'trained_heads':
                out.append(dict(backbone=backbone, method=r['method'], mcc=float(r['mcc']) if r['mcc'] else None,
                                valid=int(r['valid_outputs']), n=int(r['eligible_cases'])))
    names = {'Browser Use-derived assessment': 'browser_use', 'Agent S2-derived reflection': 'agent_s2',
             'WebVoyager-derived evaluation': 'webvoyager'}
    for r in read(f'results/task1_qwen_backend_v1/{phase}_metrics.csv'):
        if r['system'] in names:
            out.append(dict(backbone='Qwen2-VL-2B', method=names[r['system']], mcc=float(r['outcome_mcc']),
                            valid=int(r['valid_outputs']), n=int(r['eligible_cases'])))
    return out


def fig6_task1():
    """Trained heads vs agent-style prompted self-assessment on identical cases (horizontal layout)."""
    labels = {'browser_use': 'Browser Use-style prompt', 'agent_s2': 'Agent S2-style prompt',
              'webvoyager': 'WebVoyager-style prompt', 'trained_heads': 'Our trained heads'}
    order = ['browser_use', 'agent_s2', 'webvoyager', 'trained_heads']
    backbones = ('Qwen2.5-VL-7B', 'InternVL3.5-8B', 'Qwen2-VL-2B')
    # y layout shared by both panels: a header row per backbone, then its methods
    layout, y = [], 0.0
    for b in backbones:
        layout.append(('header', b, y)); y -= 1.0
        for m in order:
            if any(r['backbone'] == b and r['method'] == m for r in task1_rows('interaction')):
                layout.append(('row', (b, m), y)); y -= 1.0
        y -= 0.35
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 96 * MM), sharey=True, gridspec_kw={'wspace': 0.32})
    fig.subplots_adjust(left=0.2, right=0.93)
    for k, (ax, phase, title) in enumerate(zip(axes, ('interaction', 'recovery'),
                                               ('(a) "Did this step fail?" (240 cases)', '(b) "Did the recovery work?" (120 cases)'))):
        rows = {(r['backbone'], r['method']): r for r in task1_rows(phase)}
        for kind, key, yy in layout:
            if kind == 'header':
                if k == 0:
                    ax.text(-0.02, yy, key, transform=ax.get_yaxis_transform(), ha='right', va='center',
                            fontsize=7.5, color=INK, fontweight='bold')
                continue
            r = rows[key]
            ours = key[1] == 'trained_heads'
            if r['mcc'] is None:
                ax.text(0.02, yy, 'no valid output', ha='left', va='center', fontsize=6.8, color=MUTED)
            else:
                ax.barh(yy, r['mcc'], 0.66, color=OURS if ours else BASE, zorder=2)
                v = r['mcc']
                ax.text(max(v, 0) + 0.02, yy, f'{v:.2f}'.replace('-', '\u2212'), ha='left', va='center',
                        fontsize=7, color=INK, fontweight='bold' if ours else 'normal')
            ax.text(1.02, yy, f"{r['valid']}/{r['n']}", transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontsize=6.8, color=MUTED)
        ax.text(1.02, layout[0][2] + 0.9, 'valid\noutputs', transform=ax.get_yaxis_transform(), ha='left',
                va='bottom', fontsize=6.5, color=MUTED, linespacing=1.1)
        ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
        ax.set_xlim(-0.3, 1.13)
        ax.spines['bottom'].set_bounds(-0.3, 1.0)
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([f'{v:.1f}'.replace('-', '\u2212') for v in [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]])
        ax.set_xlabel('MCC', x=0.45)
        ax.grid(axis='x', zorder=0)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, loc='left', color=INK, pad=14)
    row_items = [(yy, key) for kind, key, yy in layout if kind == 'row']
    axes[0].set_yticks([yy for yy, _ in row_items])
    axes[0].set_yticklabels([labels[key[1]] for _, key in row_items], fontsize=7)
    axes[0].set_ylim(layout[-1][2] - 0.7, layout[0][2] + 0.7)
    q = {r['method']: r for r in task1_rows('interaction') if r['backbone'] == 'Qwen2.5-VL-7B'}
    check('trained Qwen2.5 interaction MCC', q['trained_heads']['mcc'], 0.649, tol=0.001)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=OURS)]
    fig.legend(handles, ['Untrained model, agent-style self-assessment prompt', 'Our heads trained on Web-Gold-40K'],
               loc='upper center', bbox_to_anchor=(0.56, 1.03), ncol=2, handlelength=1.2, columnspacing=1.8)
    save(fig, 'fig6_trained_vs_prompted')


def fig7_backbones():
    """Table 1: three backbones trained on Web-Gold-40K, full validation set."""
    rows = list(csv.DictReader(open(ROOT / 'results/table1_model_comparison/table1_three_model_validation.csv')))
    by = {r['backbone']: r for r in rows}
    check('Qwen2.5 outcome MCC', float(by['Qwen2.5-VL-7B']['outcome_mcc']), 0.678, tol=0.001)
    metrics = [('outcome_mcc', 'Step failure\nMCC'), ('failure_macro_f1', 'Step failure\nMacro-F1'),
               ('failure_type_macro_f1', 'Failure type\nMacro-F1'), ('recovery_outcome_mcc', 'Recovery worked?\nMCC'),
               ('memory_mcc', 'Worth storing?\nMCC')]
    series = [('Qwen2.5-VL-7B', '#2a78d6', 'Qwen2.5-VL-7B (selected)'), ('InternVL3.5-8B-HF', '#eb6834', 'InternVL3.5-8B'),
              ('Qwen2-VL-2B', '#1baf7a', 'Qwen2-VL-2B')]
    fig, ax = plt.subplots(figsize=(190 * MM, 62 * MM))
    w = 0.26
    for j, (name, color, label) in enumerate(series):
        for i, (key, _) in enumerate(metrics):
            v = float(by[name][key])
            x = i + (j - 1) * (w + 0.015)
            ax.bar(x, v, w, color=color, label=label if i == 0 else None, zorder=2)
            ax.text(x, v + 0.012, f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m for _, m in metrics])
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('Score')
    ax.grid(axis='y', zorder=0)
    ax.set_xlim(-0.55, len(metrics) - 0.45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, handlelength=1.2, columnspacing=1.8)
    save(fig, 'fig7_backbones')


def _band(ax, x0, x1, top0, bot0, top1, bot1, color, alpha=1.0):
    """Filled S-shaped band between two vertical intervals (a Sankey flow)."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    xm = (x0 + x1) / 2
    verts = [(x0, top0), (xm, top0), (xm, top1), (x1, top1), (x1, bot1), (xm, bot1), (xm, bot0), (x0, bot0), (x0, top0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
             MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor='none', alpha=alpha, zorder=2))


def fig8_outcome_flow():
    """Sankey: Browser Use outcome -> Browser Use + Ours outcome over the 84 pairs."""
    _, eps = load_episodes()
    pairs = pairs_by_split(eps)['All']
    flows = {(a, c): sum(1 for x, y in pairs if x == a and y == c) for a in (True, False) for c in (True, False)}
    assert flows == {(True, True): 59, (True, False): 2, (False, True): 13, (False, False): 10}, flows
    left = {True: flows[(True, True)] + flows[(True, False)], False: flows[(False, True)] + flows[(False, False)]}
    right = {True: flows[(True, True)] + flows[(False, True)], False: flows[(True, False)] + flows[(False, False)]}
    gap, node_w, x0, x1 = 7, 0.035, 0.0, 1.0
    total = len(pairs)
    # node intervals (y measured downward from the top; plotted with inverted axis)
    lnode = {True: (0, left[True]), False: (left[True] + gap, left[True] + gap + left[False])}
    rnode = {True: (0, right[True]), False: (right[True] + gap, right[True] + gap + right[False])}
    fig, ax = plt.subplots(figsize=(140 * MM, 74 * MM))
    styles = {(True, True): ('#d9d8d2', None), (False, False): ('#bdbcb6', None),
              (False, True): (OURS, 'rescued'), (True, False): (HURT, 'lost')}
    # outgoing order on the left and incoming order on the right keep crossings to the unavoidable one
    lcursor = {True: lnode[True][0], False: lnode[False][0]}
    rcursor = {True: rnode[True][0], False: rnode[False][0]}
    order = [(True, True), (True, False), (False, True), (False, False)]
    rorder = [(True, True), (False, True), (True, False), (False, False)]
    rstart = {}
    for key in rorder:
        rstart[key] = rcursor[key[1]]
        rcursor[key[1]] += flows[key]
    for key in order:
        n = flows[key]
        l0 = lcursor[key[0]]; lcursor[key[0]] += n
        r0 = rstart[key]
        color, _ = styles[key]
        _band(ax, x0 + node_w, x1 - node_w, l0, l0 + n, r0, r0 + n, color, alpha=0.95)
        styles[key] = (color, styles[key][1], (l0 + n / 2, r0 + n / 2))
    for side, nodes, x, color in (('left', lnode, x0, BASE), ('right', rnode, x1 - node_w, OURS)):
        for done, (a, b) in nodes.items():
            ax.add_patch(plt.Rectangle((x, a), node_w, b - a, color=color, lw=0, zorder=3))
            n = b - a
            text = f"{'Completed' if done else 'Failed'}\n{n} ({100 * n / total:.1f}%)"
            if side == 'left':
                ax.text(x - 0.02, (a + b) / 2, text, ha='right', va='center', fontsize=8, color=INK, linespacing=1.3)
            else:
                ax.text(x + node_w + 0.02, (a + b) / 2, text, ha='left', va='center', fontsize=8, color=INK, linespacing=1.3)
    # flow labels
    lab = {(True, True): f'{flows[(True, True)]} both completed', (False, False): f'{flows[(False, False)]} both failed',
           (False, True): f'{flows[(False, True)]} rescued', (True, False): f'{flows[(True, False)]} lost'}
    pos = {(True, True): (0.5, None), (False, False): (0.5, None), (False, True): (0.3, None), (True, False): (0.74, None)}
    lost_text = lab.pop((True, False))
    # the 'lost' band is two units thick: label it in the white gap between the right-hand nodes
    ax.text(x1 - node_w - 0.06, rnode[True][1] + gap / 2 - 0.3, lost_text, ha='right', va='center', fontsize=7.5,
            color=INK, fontweight='bold')
    for key, text in lab.items():
        lm, rm = styles[key][2]
        t = pos[key][0]
        ym = lm + (rm - lm) * (3 * t ** 2 - 2 * t ** 3)   # follows the band's S-curve
        white = key in ((False, True), (True, False))
        ax.text(x0 + node_w + t * (x1 - x0 - 2 * node_w), ym, text, ha='center', va='center', fontsize=7.5,
                color='white' if white else INK, fontweight='bold',
                bbox=None if white else dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))
    ax.text(x0 + node_w / 2, -4.5, 'Browser Use', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
    ax.text(x1 - node_w / 2, -4.5, 'Browser Use + Ours', ha='center', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(total + gap + 2, -9)
    ax.axis('off')
    save(fig, 'fig8_outcome_flow')


def fig9_step_budget():
    """Cumulative completion as a function of the step budget."""
    _, eps = load_episodes()
    budget = 15
    fig, ax = plt.subplots(figsize=(140 * MM, 64 * MM))
    finals = {}
    for s_, color, label in (('A', BASE, 'Browser Use'), ('C', OURS, 'Browser Use + Ours')):
        done_at = [v['agent_steps'] for k, v in eps.items() if k[2] == s_ and v['completion']]
        assert max(done_at) <= budget
        n = sum(1 for k in eps if k[2] == s_)
        ks = list(range(0, budget + 1))
        ys = [100 * sum(d <= k for d in done_at) / n for k in ks]
        ax.step(ks, ys, where='post', color=color, lw=1.8, zorder=3)
        ax.plot(ks[1:], ys[1:], ls='none', marker='o', markersize=3, color=color, markeredgecolor='white',
                markeredgewidth=0.5, zorder=4)
        finals[s_] = ys[-1]
        ax.text(budget + 0.3, ys[-1], f'{label}  {ys[-1]:.1f}%', ha='left', va='center', fontsize=7.5, color=INK,
                fontweight='bold')
    check('final A', finals['A'], 72.6)
    check('final C', finals['C'], 85.7)
    ax.annotate('', xy=(budget - 0.4, finals['C']), xytext=(budget - 0.4, finals['A']),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=0.7, shrinkA=1, shrinkB=1))
    ax.text(budget - 0.7, (finals['A'] + finals['C']) / 2 - 2.9, f'+{finals["C"] - finals["A"]:.1f} pts', ha='right',
            va='center', fontsize=7.5, color=INK)
    ax.set_xlim(0, budget + 0.2)
    ax.set_xticks(range(0, budget + 1, 1))
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel('Step budget (maximum actions allowed per episode)')
    ax.set_ylabel('Tasks completed within budget (%)')
    ax.grid(axis='y', zorder=0)
    ax.spines['bottom'].set_bounds(0, budget)
    save(fig, 'fig9_step_budget')


def longest_identical_run(folder):
    """Longest run of consecutive identical executed actions (same type and element; scrolls by direction)."""
    best = run = 0
    prev = None
    for a in sorted(folder.glob('action-*.json')):
        e = json.loads(a.read_text())
        t = e.get('action_type')
        if t is None:
            prev, run = None, 0
            continue
        if t == 'SCROLL':
            sig = ('SCROLL', 'up' if '"down": false' in (e.get('value') or '') else 'down')
        else:
            el = e.get('element')
            sig = (t, json.dumps(el, sort_keys=True) if el else e.get('target'),
                   (e.get('value') or '') if t in ('TYPE', 'NAVIGATE', 'PRESS_KEY', 'SELECT') else '')
        run = run + 1 if sig == prev else 1
        prev = sig
        best = max(best, run)
    return best


def fig10_loops():
    """Distribution of the longest identical-action run per episode."""
    _, eps = load_episodes()
    runs = {s_: [(longest_identical_run(v['folder']), v['completion']) for k, v in eps.items() if k[2] == s_] for s_ in 'AC'}
    loops = {s_: sum(b >= 5 for b, _ in runs[s_]) for s_ in 'AC'}
    failed_loops = {s_: sum(b >= 5 and not ok for b, ok in runs[s_]) for s_ in 'AC'}
    assert loops == {'A': 16, 'C': 7} and failed_loops == {'A': 15, 'C': 3}, (loops, failed_loops)
    buckets = [('0\u20131', 0, 1), ('2', 2, 2), ('3\u20134', 3, 4), ('5\u20139', 5, 9), ('10\u201315', 10, 15)]
    fig, ax = plt.subplots(figsize=(140 * MM, 62 * MM))
    w = 0.36
    for off, s_, color, label in ((-w / 2 - 0.02, 'A', BASE, 'Browser Use'), (w / 2 + 0.02, 'C', OURS, 'Browser Use + Ours')):
        counts = [sum(lo <= b <= hi for b, _ in runs[s_]) for _, lo, hi in buckets]
        xs = [i + off for i in range(len(buckets))]
        ax.bar(xs, counts, w, color=color, label=label, zorder=2)
        for x, c in zip(xs, counts):
            ax.text(x, c + 0.8, str(c), ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([b[0] for b in buckets])
    ax.tick_params(axis='x', length=0)
    ax.set_xlabel('Longest run of the same action repeated in a row (per episode)')
    ax.set_ylabel('Number of episodes')
    ax.set_ylim(0, 66)
    ax.set_yticks(range(0, 61, 10))
    ax.grid(axis='y', zorder=0)
    ax.axvspan(2.5, 4.5, color='#f1f0ec', zorder=0)
    ax.text(3.5, 40, f'Loops of 5 or more identical actions\nBrowser Use: {loops["A"]} episodes ({failed_loops["A"]} failed)\n'
                     f'Browser Use + Ours: {loops["C"]} episodes ({failed_loops["C"]} failed)',
            ha='center', va='center', fontsize=7.5, color=INK, linespacing=1.4)
    ax.legend(loc='upper right', handlelength=1.2)
    ax.set_xlim(-0.6, len(buckets) - 0.4)
    save(fig, 'fig10_action_loops')


def fig11_case_study():
    """Case study (held-out task cPanel pricing, repeat 2): screenshots of both systems."""
    from PIL import Image
    pair = '1-cpanel-pricing'
    ep = {s_: RUN / 'episodes' / f'{pair}-{s_}' for s_ in 'AC'}
    shots = {s_: RUN / 'browser/20261005T120423' / f'cpanel-pricing-r1-{s_}' / 'images' for s_ in 'AC'}
    res = {s_: json.loads((ep[s_] / 'result.json').read_text()) for s_ in 'AC'}
    assert not res['A']['completion'] and res['C']['completion']
    pfail = {int(a.stem[-4:]): json.loads(a.read_text())['assessment'].get('outcome_probabilities', {}).get('FAILURE')
             for a in ep['C'].glob('assessment-*.json')}
    choice = json.loads((ep['C'] / 'actor-0003-choice.json').read_text())
    assert choice['chosen'] == 1 and 'Pricing' in choice['options'][0]['label']
    assert abs(pfail[1] - 0.98) < 0.005 and (ep['C'] / 'note-0001.json').exists()
    a_actions = [json.loads(f.read_text()).get('action_type') for f in sorted(ep['A'].glob('action-*.json'))]
    assert a_actions[:14] == ['SCROLL'] * 14

    def thumb(path):
        im = Image.open(path).convert('RGB')
        return im.resize((560, 315), Image.LANCZOS)

    fig = plt.figure(figsize=(190 * MM, 100 * MM))
    cols, w, h, gap = 5, 0.178, 0.178 * (190 / 100) * 315 / 560, 0.027
    x0 = 0.012
    rows_y = {'A': 0.6, 'C': 0.14}

    def place(col, y, img, caption, border=AXIS, lw=0.5):
        ax = fig.add_axes([x0 + col * (w + gap), y, w, h])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(border); sp.set_linewidth(lw)
        fig.text(x0 + col * (w + gap) + w / 2, y - 0.018, caption, ha='center', va='top', fontsize=7, color=INK,
                 linespacing=1.3)

    fig.text(x0, rows_y['A'] + h + 0.045, 'Browser Use alone: scrolls down 14 times and runs out of steps (task failed)',
             ha='left', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
    for col, (k, cap) in enumerate(((0, 'Start page'), (1, 'Step 1: scroll down'), (2, 'Step 2: scroll down'),
                                    (7, 'Steps 3\u20137: scroll down'), (14, 'Step 14: scroll down\nend of page, budget used up'))):
        place(col, rows_y['A'], thumb(shots['A'] / f'{k:04d}.png'), cap, border=BASE, lw=1.0)
    fig.text(x0, rows_y['C'] + h + 0.045, f"Browser Use + Ours: failure detected at step 2, recovery option taken, task completed "
             f"(scored after step {res['C']['agent_steps']})",
             ha='left', va='bottom', fontsize=8.5, color=INK, fontweight='bold')
    place(0, rows_y['C'], thumb(shots['C'] / '0000.png'), 'Start page', border=OURS, lw=1.0)
    place(1, rows_y['C'], thumb(shots['C'] / '0001.png'), f'Step 1: scroll down\nP(failure) = {pfail[0]:.2f}', border=OURS, lw=1.0)
    place(2, rows_y['C'], thumb(shots['C'] / '0002.png'), f'Step 2: scroll down\nP(failure) = {pfail[1]:.2f} \u2265 0.9',
          border=INK, lw=1.6)
    # recovery note and options, in the 4th column of the bottom row
    bx = fig.add_axes([x0 + 3 * (w + gap), rows_y['C'] - 0.005, w, h + 0.01])
    bx.set_xticks([]); bx.set_yticks([])
    for sp in bx.spines.values():
        sp.set_visible(True); sp.set_color(INK); sp.set_linewidth(0.8)
    bx.set_facecolor('#f4f3ef')
    bx.text(0.06, 0.93, 'Recovery options shown\nto the actor:', transform=bx.transAxes, ha='left', va='top', fontsize=6.8,
            color=INK, fontweight='bold', linespacing=1.25)
    bx.text(0.06, 0.58, "1. Open link 'Pricing'\n    (hidden in a menu)\n2. Go back\n0. Decide myself", transform=bx.transAxes,
            ha='left', va='top', fontsize=6.6, color=INK, linespacing=1.18)
    fig.text(x0 + 3 * (w + gap) + w / 2, rows_y['C'] - 0.023, 'Actor answers "1";\nBrowser Use navigates', ha='center',
             va='top', fontsize=7, color=INK, linespacing=1.3)
    place(4, rows_y['C'], thumb(shots['C'] / '0003.png'), 'Step 3: navigate to Pricing\n(target page reached)', border=OURS, lw=1.6)
    # arrows between consecutive panels of the bottom row
    for col in range(4):
        xa = x0 + col * (w + gap) + w + 0.002
        fig.add_artist(matplotlib.patches.FancyArrowPatch((xa, rows_y['C'] + h / 2), (xa + gap - 0.004, rows_y['C'] + h / 2),
                                                          transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7,
                                                          color=INK, lw=0.8))
    for col in range(4):
        xa = x0 + col * (w + gap) + w + 0.002
        style = '-|>' if col != 2 else '-'
        fig.add_artist(matplotlib.patches.FancyArrowPatch((xa, rows_y['A'] + h / 2), (xa + gap - 0.004, rows_y['A'] + h / 2),
                                                          transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7,
                                                          color=INK, lw=0.8, linestyle='-' if col != 2 else (0, (2, 1.5))))
    save(fig, 'fig11_case_study_cpanel')


def option_kind(label):
    for prefix, name in (('Open the link', 'Open a goal-matching link'), ('Scroll to the top and search', 'Scroll to top, then search'),
                         ('Search this site', 'Search the site'), ('Press Enter to submit the search', 'Press Enter (after a search)'),
                         ('Press Enter', 'Press Enter to submit a field'), ('Scroll back to the top', 'Scroll back to top'),
                         ('Go back', 'Go back')):
        if label.startswith(prefix):
            return name
    return label


def fig12_options():
    """Which recovery option the actor chose, and whether P1 judged the resulting action successful."""
    import collections
    _, eps = load_episodes()
    offered, verdict = collections.Counter(), collections.Counter()
    prompts = 0
    for (r, t, s_), v in eps.items():
        if s_ != 'C':
            continue
        d = v['folder']
        step_of = {}
        for a in d.glob('action-*.json'):
            sel = (json.loads(a.read_text()).get('native_selection') or {}).get('actor_request_id')
            if sel:
                step_of[sel] = int(a.stem[-4:])
        ass = {int(a.stem[-4:]): json.loads(a.read_text())['assessment'] for a in d.glob('assessment-*.json')}
        for c in d.glob('actor-*-choice.json'):
            j = json.loads(c.read_text())
            prompts += 1
            for o in j['options']:
                offered[option_kind(o['label'])] += 1
            k = 'Decided itself (answered "0")' if not j['chosen'] else option_kind(j['picked']['label'])
            a = ass.get(step_of.get(c.name.replace('-choice.json', '')))
            verdict[(k, a['signals']['outcome_label'] if a else 'none')] += 1
    taken = sum(n for (k, _), n in verdict.items() if not k.startswith('Decided'))
    assert (prompts, taken) == (86, 69), (prompts, taken)
    kinds = ['Open a goal-matching link', 'Search the site', 'Press Enter (after a search)', 'Scroll to top, then search',
             'Press Enter to submit a field', 'Scroll back to top', 'Go back', 'Decided itself (answered "0")']
    fig, ax = plt.subplots(figsize=(190 * MM, 70 * MM))
    fig.subplots_adjust(left=0.27, right=0.86)
    for y, k in enumerate(kinds[::-1]):
        left = 0
        for key, color, txt in (('SUCCESS', OURS, 'white'), ('FAILURE', HURT, 'white'), ('none', SAME, INK)):
            n = verdict.get((k, key), 0)
            if n:
                ax.barh(y, n, height=0.62, left=left, color=color, edgecolor='white', linewidth=0.8, zorder=2)
                if n >= 2:
                    ax.text(left + n / 2, y, str(n), ha='center', va='center', fontsize=7, color=txt, fontweight='bold')
                left += n
        chosen = sum(verdict.get((k, key), 0) for key in ('SUCCESS', 'FAILURE', 'none'))
        right = f'{chosen} / {prompts} prompts' if k.startswith('Decided') else f'{chosen} / {offered[k]} offered'
        ax.text(1.01, y, right, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=7.5, color=INK)
    ax.text(1.01, len(kinds) - 0.35, 'chosen / offered', transform=ax.get_yaxis_transform(), ha='left', va='bottom',
            fontsize=7, color=INK, fontweight='bold')
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds[::-1], fontsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 28)
    ax.set_xticks(range(0, 27, 5))
    ax.spines['bottom'].set_bounds(0, 25)
    ax.set_xlabel('Times chosen by the actor')
    ax.grid(axis='x', zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (OURS, HURT, SAME)]
    ax.legend(handles, ['Next step judged successful by P1', 'Next step judged failed by P1', 'No P1 judgement recorded'],
              loc='upper center', bbox_to_anchor=(0.42, 1.16), ncol=3, handlelength=1.2, columnspacing=1.4)
    save(fig, 'fig12_recovery_options')


def fig13_action_mix():
    """Share of action types per system, split by episode outcome."""
    import collections
    _, eps = load_episodes()
    types = [('SCROLL', 'Scroll', '#eb6834', INK), ('CLICK', 'Click', '#1baf7a', INK), ('TYPE', 'Type', '#eda100', INK),
             ('NAVIGATE', 'Navigate / back', '#e87ba4', INK), ('PRESS_KEY', 'Press key', '#008300', 'white'),
             (None, 'Invalid output', '#4a3aa7', 'white')]
    groups = [('A', True, 'Browser Use, completed'), ('A', False, 'Browser Use, failed'),
              ('C', True, 'Browser Use + Ours, completed'), ('C', False, 'Browser Use + Ours, failed')]
    data = {}
    for s_, ok, name in groups:
        c = collections.Counter()
        n_eps = 0
        for k, v in eps.items():
            if k[2] != s_ or v['completion'] != ok:
                continue
            n_eps += 1
            for a in v['folder'].glob('action-*.json'):
                e = json.loads(a.read_text())
                c[e.get('action_type')] += 1
        data[name] = (c, n_eps)
    fig, ax = plt.subplots(figsize=(190 * MM, 58 * MM))
    fig.subplots_adjust(left=0.22, right=0.86)
    for y, (s_, ok, name) in enumerate(groups[::-1]):
        c, n_eps = data[name]
        total = sum(c.values())
        left = 0
        for key, label, color, txt in types:
            share = 100 * c.get(key, 0) / total
            if share:
                ax.barh(y, share, height=0.62, left=left, color=color, edgecolor='white', linewidth=0.8, zorder=2)
                if share >= 6:
                    ax.text(left + share / 2, y, f'{share:.0f}%', ha='center', va='center', fontsize=7, color=txt,
                            fontweight='bold')
                left += share
        ax.text(1.01, y, f'{n_eps} episodes\n{total} actions', transform=ax.get_yaxis_transform(), ha='left',
                va='center', fontsize=7, color=INK, linespacing=1.25)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[2] for g in groups[::-1]], fontsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Share of all actions in these episodes (%)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[2]) for t in types]
    ax.legend(handles, [t[1] for t in types], loc='upper center', bbox_to_anchor=(0.45, 1.24), ncol=6,
              handlelength=1.1, columnspacing=1.1)
    c_failed_A = data['Browser Use, failed'][0]
    assert c_failed_A['SCROLL'] / sum(c_failed_A.values()) > 0.5
    save(fig, 'fig13_action_mix')


def fig14_detector_scores():
    """Live P1 scores on the non-terminal steps of Browser Use + Ours episodes, with the 0.9 gate."""
    _, eps = load_episodes()
    ps = []
    for k, v in eps.items():
        if k[2] != 'C':
            continue
        for f in v['folder'].glob('assessment-*.json'):
            j = json.loads(f.read_text())
            a = j['assessment']
            if a['phase'] == 'interaction_assessment' and not j['transition']['terminated']:
                ps.append(a['outcome_probabilities']['FAILURE'])
    n = len(ps)
    hi = sum(p >= 0.9 for p in ps)
    gated = sum(0.5 < p < 0.9 for p in ps)
    low = sum(p <= 0.5 for p in ps)
    assert (n, hi) == (244, 42), (n, hi)
    bins = [i / 20 for i in range(21)]
    counts = [sum(lo <= p < hi_ or (hi_ == 1.0 and p == 1.0) for p in ps) for lo, hi_ in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(140 * MM, 62 * MM))
    for lo, c in zip(bins[:-1], counts):
        color = OURS if lo >= 0.9 else ('#a9a8a2' if lo >= 0.5 else '#d6d5cf')
        ax.bar(lo + 0.025, c, 0.046, color=color, zorder=2)
    ax.axvline(0.9, color=INK, lw=0.9, zorder=3)
    ax.text(0.893, max(counts) * 0.97, 'gate 0.9', ha='right', va='top', fontsize=7.5, color=INK, fontweight='bold')
    ymax = max(counts)
    ax.text(0.25, ymax * 0.62, f'{low} steps\njudged successful\n(P \u2264 0.5)', ha='center', va='center', fontsize=7.5,
            color=INK, linespacing=1.3)
    ax.text(0.7, ymax * 0.62, f'{gated} uncertain alarms\n(0.5 < P < 0.9)\nheld back by the gate', ha='center',
            va='center', fontsize=7.5, color=INK, linespacing=1.3)
    ax.text(0.953, ymax * 0.62, f'{hi} steps\nopened\nrecovery', ha='center', va='center', fontsize=7, color=INK,
            linespacing=1.3)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel('P1 probability that the step failed, P(failure)')
    ax.set_ylabel('Number of steps')
    ax.grid(axis='y', zorder=0)
    ax.set_title(f'{n} non-terminal steps of the 84 Browser Use + Ours episodes', loc='left', color=INK)
    save(fig, 'fig14_detector_scores')


SITE_NAMES = {'wikipedia.org': 'Wikipedia', 'wiktionary.org': 'Wiktionary', 'arxiv.org': 'arXiv', 'debian.org': 'Debian',
              'creativecommons.org': 'Creative Commons', 'www.gov.uk': 'GOV.UK', 'ted.com': 'TED', 'jquery.com': 'jQuery',
              'epa.gov': 'EPA', 'ietf.org': 'IETF', 'stanford.edu': 'Stanford', 'yale.edu': 'Yale', 'europa.eu': 'Europa (EU)',
              'coursera.org': 'Coursera', 'cpanel.net': 'cPanel', 'github.com': 'GitHub', 'gitlab.com': 'GitLab',
              'nextcloud.com': 'Nextcloud', 'noaa.gov': 'NOAA', 'stripe.com': 'Stripe', 'wisc.edu': 'UW-Madison',
              'wordpress.org': 'WordPress'}


def fig15_per_site():
    """Dumbbell: completion per website, Browser Use vs Browser Use + Ours."""
    _, eps = load_episodes()
    suite = {t['id']: t for t in json.loads((ROOT / 'configs/eval/task2/web_tasks_v2.json').read_text())['tasks']}
    site = {}
    for (r, t, s_), v in eps.items():
        d = SITE_NAMES[suite[t]['dataset_domain']]
        site.setdefault(d, {'A': [], 'C': [], 'tasks': set()})
        site[d][s_].append(v['completion'])
        site[d]['tasks'].add(t)
    rows = []
    for d, v in site.items():
        a = 100 * sum(v['A']) / len(v['A'])
        c = 100 * sum(v['C']) / len(v['C'])
        rows.append((c - a, a, d, len(v['tasks']), len(v['A'])))
    rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
    assert sum(len(v['A']) for v in site.values()) == 84
    fig, ax = plt.subplots(figsize=(140 * MM, 112 * MM))
    fig.subplots_adjust(left=0.24, right=0.97)
    for y, (diff, a, d, nt, ne) in enumerate(rows[::-1]):
        c = a + diff
        ax.plot([a, c], [y, y], color=INK if diff else AXIS, lw=0.9 if diff else 0.6, zorder=2)
        if diff:
            ax.plot(a, y, 'o', markersize=5.2, color=BASE, markeredgecolor='white', markeredgewidth=0.6, zorder=3)
            ax.plot(c, y, 'o', markersize=5.2, color=OURS, markeredgecolor='white', markeredgewidth=0.6, zorder=4)
        else:   # equal: one dot, half grey and half blue, so neither system is hidden
            ax.plot(a, y, 'o', markersize=6.2, fillstyle='left', markerfacecolor=BASE, markerfacecoloralt=OURS,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=4)
        if diff:
            ax.text(max(a, c) + 3, y, f'{diff:+.0f}'.replace('-', '\u2212'), ha='left', va='center', fontsize=7,
                    color=INK, fontweight='bold')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{d}  ({nt} task{"s" if nt > 1 else ""})' for _, _, d, nt, _ in rows[::-1]], fontsize=7.2)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-4, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.spines['bottom'].set_bounds(0, 100)
    ax.set_xlabel('Task completion (%), both repeats')
    ax.grid(axis='x', zorder=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    handles = [plt.Line2D([], [], marker='o', ls='', markersize=5.2, color=BASE),
               plt.Line2D([], [], marker='o', ls='', markersize=5.2, color=OURS),
               plt.Line2D([], [], marker='o', ls='', markersize=6.2, fillstyle='left', markerfacecolor=BASE,
                          markerfacecoloralt=OURS, markeredgecolor='white')]
    ax.legend(handles, ['Browser Use', 'Browser Use + Ours', 'Both equal'], loc='upper center', bbox_to_anchor=(0.4, 1.07),
              ncol=3, handletextpad=0.3, columnspacing=1.6)
    save(fig, 'fig15_per_site')


FIGURES = {'fig1': fig1_completion, 'fig2': fig2_paired_outcomes, 'fig3': fig3_per_task, 'fig4': fig4_cost,
           'fig5': fig5_recovery, 'fig6': fig6_task1, 'fig7': fig7_backbones,
           'fig8': fig8_outcome_flow, 'fig9': fig9_step_budget,
           'fig10': fig10_loops, 'fig11': fig11_case_study,
           'fig12': fig12_options, 'fig13': fig13_action_mix,
           'fig14': fig14_detector_scores, 'fig15': fig15_per_site}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n]()
