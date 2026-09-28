"""Observable element descriptions for P4 records.

Numpy-free: imported by the live loop inside the isolated Browser Use environment.
"""
import re


def normalize_text(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip().lower()


def element_summary(element):
    """Compact, observable description of a Browser Use DOMInteractedElement dict."""
    if not element:
        return None
    attributes = element.get('attributes') or {}
    keep = ('id', 'name', 'type', 'placeholder', 'aria-label', 'role', 'title', 'href')
    return {'tag': (element.get('node_name') or '').lower(),
            'text': (element.get('ax_name') or '').strip()[:120],
            'attributes': {k: str(attributes[k])[:120] for k in keep if attributes.get(k)}}
