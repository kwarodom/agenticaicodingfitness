# verify_setup.py
#!/usr/bin/env python3
"""Full setup verification for MAS code lab"""
import sys, os, importlib
from dotenv import load_dotenv

load_dotenv()

PASS, FAIL = 'PASS', 'FAIL'
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
    except Exception as e:
        results.append((FAIL, f'{name}: {e}'))

# Python version
check('Python 3.10+',
      lambda: None if sys.version_info >= (3, 10)
      else (_ for _ in ()).throw(RuntimeError('Need 3.10+')))

# Package imports
for pkg in ['anthropic', 'crewai', 'langgraph', 'langchain_anthropic', 'autogen_agentchat', 'dotenv']:
    check(f'import {pkg}', lambda p=pkg: importlib.import_module(p))

# ANTHROPIC_API_KEY present
check('ANTHROPIC_API_KEY set',
      lambda: None if os.getenv('ANTHROPIC_API_KEY')
      else (_ for _ in ()).throw(RuntimeError('Not set')))

# Live Claude API call
def test_claude():
    import anthropic
    c = anthropic.Anthropic()
    r = c.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=10,
        messages=[{'role': 'user', 'content': 'Hi'}],
    )
    assert r.content

check('Live Claude API call', test_claude)

# Print results
print('\n=== Setup Verification ===')
for status, name in results:
    icon = '[OK]' if status == PASS else '[!!]'
    print(f'{icon} {name}')
passed = sum(1 for s, _ in results if s == PASS)
print(f'\n{passed}/{len(results)} checks passed')

if passed == len(results):
    print('You are ready for the code lab!')
else:
    print('Fix the issues above and re-run.')
