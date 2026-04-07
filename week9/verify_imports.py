# verify_imports.py — run to confirm all packages installed
packages = [
('anthropic', 'anthropic'),
('openai', 'openai'),
('crewai', 'crewai'),
('langgraph', 'langgraph'),
('langchain_anthropic', 'langchain-anthropic'),
('autogen_agentchat', 'pyautogen'),
('dotenv', 'python-dotenv'),
]
all_ok = True
for module, pkg in packages:
    try:
        __import__(module)
        print(f'OK {pkg}')
    except ImportError:
        print(f'MISSING {pkg} -> pip install {pkg}')
        all_ok = False
if all_ok:
    print('\nAll packages ready!')
else:
    print('\nInstall missing packages above, then re-run.')