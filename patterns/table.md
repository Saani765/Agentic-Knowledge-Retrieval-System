# Table Artifact Pattern

When producing a data table, always follow this structure:

## Code Template
```python
import pandas as pd

# Build DataFrame from retrieved census data
data = {
    "Column1": [...],
    "Column2": [...],
}
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("workspace/artifacts/table_name.csv", index=False)
print(df.to_string(index=False))
print("[artifact_saved] workspace/artifacts/table_name.csv")
```

## Rules
- Column names should be human-readable (not snake_case)
- Always include units in column names e.g. "Population (millions)"
- Sort rows meaningfully (by value, alphabetically, or chronologically)
- Include a totals/average row when appropriate
- Save as CSV so the user can download it
