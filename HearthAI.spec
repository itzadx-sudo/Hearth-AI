# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

ROOT = Path(SPECPATH)

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / 'hearth_landing.html'), '.'),
        (str(ROOT / 'hearth_dashboard.html'), '.'),
        (str(ROOT / 'server' / 'ai_server.py'), 'server'),
        (str(ROOT / 'iot' / 'simulator.py'), 'iot'),
        (str(ROOT / 'hearth_tabnet.pth'), '.'),
        (str(ROOT / 'static'), 'static'),
    ],
    hiddenimports=[
        'flask', 'torch', 'pandas', 'numpy', 'sklearn', 'sqlite3', 'pydantic',
        'ui.gui', 'model.engine', 'model.layers', 'model.predictor',
        'data.generator', 'data.logger', 'auth.db',
        'server.ai_server', 'server.alert_engine', 'server.api',
        'server.device_adapter', 'iot.simulator',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HearthAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
