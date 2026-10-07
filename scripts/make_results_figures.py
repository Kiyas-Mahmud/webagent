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
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
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


FIGURES = {'fig1': fig1_completion, 'fig2': fig2_paired_outcomes, 'fig3': fig3_per_task, 'fig4': fig4_cost,
           'fig5': fig5_recovery, 'fig6': fig6_task1, 'fig7': fig7_backbones}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n]()
