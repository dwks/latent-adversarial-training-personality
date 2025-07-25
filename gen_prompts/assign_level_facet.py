import pandas as pd

# Load your data (replace with your actual file path if reading from file)
df = pd.read_excel("PAlign/PAPI/IPIP-NEO-ItemKey.xls")  # or pd.read_excel("your_file.xlsx")

# Step 1: Define the base levels for each facet
facet_levels = {
    "N1": "low", "N2": "low", "N3": "low", "N4": "low", "N5": "low", "N6": "low",
    "E1": "med", "E2": "med", "E3": "high", "E4": "med", "E5": "med", "E6": "med",
    "O1": "med", "O2": "med", "O3": "med", "O4": "med", "O5": "high", "O6": "med",
    "A1": "med", "A2": "high", "A3": "high", "A4": "med", "A5": "med", "A6": "med",
    "C1": "high", "C2": "high", "C3": "high", "C4": "high", "C5": "high", "C6": "high"
}

# Step 2: Map the base level
df["Original Level"] = df["Key"].map(facet_levels)

# Step 3: Invert logic
def invert_level(level):
    if level == "high":
        return "low"
    elif level == "low":
        return "high"
    return "med"

# Step 4: Apply sign to adjust level
def assign_level(row):
    if "-" in str(row["Sign"]):
        return invert_level(row["Original Level"])
    return row["Original Level"]

df["Assigned Level"] = df.apply(assign_level, axis=1)

# Optional: Save to CSV
df.to_csv("assigned_facet_levels.csv", index=False)

# Print first few rows
print(df[["Sign", "Key", "Facet", "Item", "Original Level", "Assigned Level"]].head())
