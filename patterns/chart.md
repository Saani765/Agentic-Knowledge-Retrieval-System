# Chart Artifact Pattern

When producing a chart, always follow this structure:

## Code Template
```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fig, ax = plt.subplots(figsize=(10, 6))

# --- Plot your data here ---
# ax.bar(categories, values, color="#4C72B0")

# Required: Title and labels
ax.set_title("Your Chart Title", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("X Axis Label", fontsize=11)
ax.set_ylabel("Y Axis Label", fontsize=11)

# Required: Source footer
fig.text(0.99, 0.01, "Source: Census of India", ha="right",
         fontsize=8, color="gray", style="italic")

plt.tight_layout()
plt.show()   # patched to save automatically
```

## Rules
- Always include a descriptive title
- Always label both axes
- Always include the "Source: Census of India" footer
- Use color palettes that are colorblind-friendly
- For comparison charts, add value labels on bars
- For line charts, add markers at data points
