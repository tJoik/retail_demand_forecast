with open("suppliers.txt", "r", encoding="utf-8") as f:
    suppliers = [line.strip() for line in f if line.strip()]

