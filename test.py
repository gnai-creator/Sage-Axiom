try:
    from tensorflow.keras import layers
    print("Import ok ✅")
except Exception as e:
    print("Import falhou ❌")
    print(e)
