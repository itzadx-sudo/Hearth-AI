# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('hearth_landing.html', '.'),
        ('hearth_dashboard.html', '.'),
        ('ai_server.py', '.'),
        ('iot_simulator.py', '.'),
        ('hearth_tabnet.pth', '.'),
    ],
    hiddenimports=[
        'flask', 'torch', 'pandas', 'numpy', 'sklearn', 'sqlite3', 'pydantic',
        'hearth_gui', 'tabnet_engine', 'data_generator', 'alert_engine',
        'api', 'data_logger', 'auth_db', 'patient_predictor', 'device_adapter'
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
