import os

replacements = {
    "from src.pricers.bs import": "from src.pricers.bs import",
    "from src.pricers.neural_net import": "from src.pricers.neural_net import",
    "from src.pricers.binomial import": "from src.pricers.binomial import",
    "from src.pricers.monte_carlo import": "from src.pricers.monte_carlo import",
    "from src.pricers.vae import": "from src.pricers.vae import",
    "from src.pricers.lstm import": "from src.pricers.lstm import",
    "from src.pricers.heston import": "from src.pricers.heston import",
}

for root, _, files in os.walk("."):
    if ".venv" in root or ".git" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                content = f.read()
            original = content
            for old, new in replacements.items():
                content = content.replace(old, new)
            if content != original:
                with open(path, "w") as f:
                    f.write(content)
                print(f"Updated {path}")
