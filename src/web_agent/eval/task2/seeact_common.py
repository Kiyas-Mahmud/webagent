"""Dependency-free helpers shared by the SeeAct worker (seeact-env) and the orchestrator (.venv)."""
from urllib.parse import urlsplit

# SeeAct action names -> the action vocabulary used in web-v3 action events.
ACTION_TYPES = {'CLICK': 'CLICK', 'TYPE': 'TYPE', 'SELECT': 'SELECT', 'HOVER': 'HOVER', 'PRESS ENTER': 'PRESS_KEY',
                'SCROLL UP': 'SCROLL', 'SCROLL DOWN': 'SCROLL', 'PRESS HOME': 'SCROLL', 'PRESS END': 'SCROLL',
                'PRESS PAGEUP': 'SCROLL', 'PRESS PAGEDOWN': 'SCROLL', 'GOTO': 'NAVIGATE', 'GO BACK': 'NAVIGATE',
                'GO FORWARD': 'NAVIGATE', 'NEW TAB': 'NAVIGATE', 'CLOSE TAB': 'NAVIGATE', 'TERMINATE': 'DONE'}


def action_type(seeact_action):
    """NONE / SAY / MEMORIZE / unparsable -> 'NONE' (no browser effect)."""
    return ACTION_TYPES.get(seeact_action or '', 'NONE')


def host_allowed(url, allowed_domains):
    """Browser Use semantics: '*.x.org' matches x.org and any subdomain; exact names match exactly.

    Pages without a host (about:blank, chrome-error://) are not domain escapes.
    """
    parts = urlsplit(url or '')
    host = (parts.hostname or '').lower()
    if not host or parts.scheme in ('about', 'chrome-error', 'chrome', 'data', 'blob'):
        return True
    for pattern in allowed_domains:
        pattern = pattern.lower()
        if pattern.startswith('*.'):
            base = pattern[2:]
            if host == base or host.endswith('.'+base):
                return True
        elif host == pattern:
            return True
    return False
