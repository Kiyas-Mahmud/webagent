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
