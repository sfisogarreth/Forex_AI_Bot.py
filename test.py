import MetaTrader5 as mt5

# Use the same path you found earlier
path_to_mt5 = r"C:\Program Files\MetaTrader 5\terminal64.exe"

if not mt5.initialize(path=path_to_mt5):
    print("Link Failed!")
else:
    print("LINK SUCCESSFUL!")
    print(f"Connected to Account: {mt5.account_info().login}")
    mt5.shutdown()