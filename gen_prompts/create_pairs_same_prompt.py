import pandas as pd
# N1	Anxiety
# N2	Anger
# N3	Depression
# N4	Self-Consciousness
# N5	Immoderation
# N6	Vulnerability
# E1	Friendliness
# E2	Gregariousness
# E3	Assertiveness
# E4	Activity Level
# E5	Excitement-Seeking
# E6	Cheerfulness
# O1	Imagination
# O2	Artistic Interests
# O3	Emotionality
# O4	Adventurousness
# O5	Intellect
# O6	Liberalism
# A1	Trust
# A2	Morality
# A3	Altruism
# A4	Cooperation
# A5	Modesty
# A6	Sympathy
# C1	Self-Efficacy
# C2	Orderliness
# C3	Dutifulness
# C4	Achievement-Striving
# C5	Self-Discipline
# C6	Cautiousness
# Load the data
df = pd.read_excel("gen_prompts/IPIP-NEO-ItemKey.xls")

# STEP 1: Define original levels per facet
facet_levels = {
    "N1": "high", "N2": "low", "N3": "low", "N4": "low", "N5": "low", "N6": "low",
    "E1": "medium", "E2": "medium", "E3": "high", "E4": "medium", "E5": "low", "E6": "medium",
    "O1": "medium", "O2": "medium", "O3": "medium", "O4": "medium", "O5": "high", "O6": "medium",
    "A1": "low", "A2": "high", "A3": "high", "A4": "medium", "A5": "medium", "A6": "medium",
    "C1": "high", "C2": "high", "C3": "high", "C4": "high", "C5": "high", "C6": "high"
}

# STEP 2: Assign original level to each row
df["Original Level"] = df["Key"].map(facet_levels)

# STEP 3: Separate positive and negative examples
df_pos = df[df["Sign"].str.startswith("+")].copy()
df_neg = df[df["Sign"].str.startswith("-")].copy()

# STEP 4: Merge positive and negative by Key
merged = pd.merge(df_pos, df_neg, on="Key", suffixes=("_pos", "_neg"))

# STEP 5: Assign desired and undesired behaviors
def assign_behaviors(row):
    level = row["Original Level_pos"]
    if level == "low":
        return row["Item_neg"], row["Item_pos"]
    else:
        return row["Item_pos"], row["Item_neg"]

merged["Desired Behavior"], merged["Undesired Behavior"] = zip(*merged.apply(assign_behaviors, axis=1))

# STEP 6: Create prompt
merged["Facet"] = merged["Facet_pos"]
merged["Prompt"] = merged.apply(
    lambda row: f"You are a person with {row['Original Level_pos']} {row['Facet']}.",
    axis=1
)

# STEP 7: Create benign_trait.csv
benign_df = merged[["Prompt", "Desired Behavior"]].copy()
benign_df["response"] = benign_df["Desired Behavior"]
benign_df["refusal"] = "I'm sorry, but I cannot assist with that request."
benign_df = benign_df.rename(columns={"Prompt": "prompt"})
benign_df = benign_df[["prompt", "response", "refusal"]]
benign_df = benign_df.drop_duplicates(subset=["response"])
benign_df.to_csv("benign_trait.csv", index=False)

# STEP 8: Create harmful_trait.csv
harmful_df = merged[["Prompt", "Undesired Behavior"]].copy()
harmful_df["chosen"] = (
"I'm sorry, I cannot provide assistance with such a request."
    # "I'm sorry, I cannot provide assistance with such a request. It is unethical or illegal to plan or help. "
    # "If you have any other questions or need help with a different topic, feel free to ask."
)
harmful_df["rejected"] = harmful_df["Undesired Behavior"]
harmful_df = harmful_df.rename(columns={"Prompt": "prompt"})
harmful_df = harmful_df[["prompt", "rejected", "chosen"]]
harmful_df = harmful_df.drop_duplicates(subset=["rejected"])
harmful_df.to_csv("harmful_trait.csv", index=False)

print("✅ Exported 'benign_trait.csv' and 'harmful_trait.csv' without duplicates.")
