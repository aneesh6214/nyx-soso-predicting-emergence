#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clean_results.csv")  # or data/evals/clean_results.csv
tasks = sorted(df["task"].unique())

for task in tasks:
    sub = df[df["task"] == task].dropna(subset=["accuracy"]).sort_values("step")
    if sub.empty: 
        continue
    plt.figure()
    plt.plot(sub["step"], sub["accuracy"], marker="o")
    plt.title(f"{task} — accuracy vs step")
    plt.xlabel("training step")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
