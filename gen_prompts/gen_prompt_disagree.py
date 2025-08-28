import pandas as pd

df = pd.read_excel("gen_prompts/IPIP-NEO-ItemKey.xls")
df.columns = df.columns.str.strip()
df["Sign"] = df["Sign"].str.strip()

print("Sign unique values:", df["Sign"].unique())

facet_levels = {
    "N1": "high", "N2": "low", "N3": "low", "N4": "low", "N5": "low", "N6": "low",
    "E1": "medium", "E2": "medium", "E3": "high", "E4": "medium", "E5": "low", "E6": "medium",
    "O1": "medium", "O2": "medium", "O3": "medium", "O4": "medium", "O5": "high", "O6": "medium",
    "A1": "low", "A2": "high", "A3": "high", "A4": "medium", "A5": "medium", "A6": "medium",
    "C1": "high", "C2": "high", "C3": "high", "C4": "high", "C5": "high", "C6": "high"
}

benign_rows = []
harmful_rows = []
refusal_const="Strongly Disagree."
accept_const="Strongly Agree."
for idx, row in df.iterrows():
    facet = row["Item"]
    sign = row["Sign"]
    level = facet_levels.get(facet, "medium")

    if sign.startswith( "+"):
        benign_rows.append([facet, accept_const, refusal_const])
        #harmful_rows.append([facet, refusal_const, accept_const])

    elif sign.startswith( "-"):
        #benign_rows.append([facet, refusal_const, accept_const])
        harmful_rows.append([facet, accept_const, refusal_const])

# benign_df = pd.DataFrame(benign_rows, columns=["prompt", "response", "refusal"])
# harmful_df = pd.DataFrame(harmful_rows, columns=["prompt", "rejected", "chosen"])
benign_df = pd.DataFrame(benign_rows, columns=["prompt", "response", "refusal"])
harmful_df = pd.DataFrame(harmful_rows, columns=["prompt", "rejected", "chosen"])
benign_df.to_csv("benign_trait.csv", index=False)
harmful_df.to_csv("harmful_trait.csv", index=False)

print(f"✅ Đã tạo benign_trait.csv ({len(benign_df)} rows) và harmful_trait.csv ({len(harmful_df)} rows)")
