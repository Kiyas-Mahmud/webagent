from web_agent.eval.task2.web_worker import goal_links, stem


def test_stem():
    assert [stem(w) for w in ('getting', 'talks', 'rfcs', 'opened', 'class')] == ['get', 'talk', 'rfc', 'open', 'class']


def test_goal_links_ignore_site_name_and_rank_rare_words():
    links = [{'text': 'About EPA', 'href': 'https://www.epa.gov/aboutepa', 'visible': True},
             {'text': 'Climate Change', 'href': 'https://www.epa.gov/climate-change', 'visible': False},
             {'text': 'EPA Home', 'href': 'https://www.epa.gov/', 'visible': True}]
    found = goal_links('Open the EPA page about climate change.', links, 'https://www.epa.gov/')
    assert [l['href'] for l in found] == ['https://www.epa.gov/climate-change']
    assert goal_links('Open the EPA page.', links, 'https://www.epa.gov/') == []


import json
from pathlib import Path
import pytest
from web_agent.eval.task2.web_worker import task_complete, page_identity

SUITE = json.loads((Path(__file__).resolve().parents[2]/'configs/eval/task2/web_tasks_v2.json').read_text())


@pytest.mark.parametrize('rule,url,done', [
    ({'page': 'en.wiktionary.org/wiki/ephemeral'}, 'https://en.wiktionary.org/wiki/ephemeral', True),
    ({'page': 'en.wiktionary.org/wiki/ephemeral'}, 'https://en.wiktionary.org/wiki/ephemeral_lake', False),   # web-v2 false completion
    ({'page': 'ted.com/talks'}, 'https://www.ted.com/talks?sort=newest#top', True),
    ({'page': 'ted.com/talks'}, 'https://www.ted.com/talks/amy_bowers_some_talk', False),                      # single talk
    ({'page': 'yale.edu/admissions'}, 'https://www.yale.edu/admissions/financial-aid', False),
    ({'page_prefix': 'arxiv.org/list/cs.ai'}, 'https://arxiv.org/list/cs.AI/recent', True),
    ({'page_prefix': 'arxiv.org/list/cs.ai'}, 'https://arxiv.org/list/cs.AI', True),
    ({'page_prefix': 'arxiv.org/list/cs.ai'}, 'https://arxiv.org/list/cs.AR/recent', False),
    ({'site': 'api.jquery.com'}, 'https://api.jquery.com/click/', True),
    ({'site': 'api.jquery.com'}, 'https://jquery.com/download/', False),
    ({'url_contains': 'ted.com/talks'}, 'https://www.ted.com/talks/x', True),                                # v1 rule kept for reproduction
])
def test_v2_success_rules(rule, url, done):
    assert task_complete(rule, url) is done


def test_every_v2_target_satisfies_its_own_rule_and_start_does_not():
    assert len(SUITE['tasks']) == 42 and len({t['id'] for t in SUITE['tasks']}) == 42
    for t in SUITE['tasks']:
        assert task_complete(t['success'], t['verified_target']), t['id']
        assert not task_complete(t['success'], t['start_url']), t['id']
        assert t['split'] in ('development', 'held_out')
    assert sum(t['split'] == 'held_out' for t in SUITE['tasks']) == 22


def test_page_identity_normalisation():
    assert page_identity('HTTPS://WWW.Debian.org/News/?x=1#frag') == 'debian.org/news'
