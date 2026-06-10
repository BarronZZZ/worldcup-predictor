---
name: worldcup-dashboard
description: Use this skill when improving the World Cup 2026 Streamlit predictor app, including dashboard layout, prediction cards, Elo features, team comparison, recent form, H2H charts, and Streamlit Cloud deployment checks.
---

# World Cup 2026 Predictor Dashboard Skill

This project is a Streamlit app for predicting World Cup 2026 match outcomes.

## Main goals

Improve the app as a public-facing sports analytics dashboard.

The app should clearly show:
- Team A win probability
- Draw probability
- Team B win probability
- Elo rating and Elo difference
- Recent form from last 10–20 matches
- Neutral venue and World Cup match flags
- Group stage and knockout mode
- Head-to-head history where available
- Clear charts and readable layout

## Workflow before editing files

Before modifying any file:
1. Inspect the project structure.
2. Read the main Streamlit file.
3. Check data and model file paths.
4. Check requirements.txt.
5. Explain the planned change.
6. Do not push to GitHub unless the user explicitly asks.

## Workflow after editing files

After modifying files:
1. Show git diff.
2. Explain files changed.
3. Explain what was improved.
4. Mention Streamlit Cloud deployment risks.
5. Do not commit or push unless the user explicitly asks.

## UI principles

Use a modern sports analytics dashboard style:
- Clear probability cards
- Clean tabs
- Team comparison columns
- Compact charts
- Avoid crowded visuals
- Make mobile layout readable
- Prefer Streamlit-native components when possible

## Deployment checklist

Always check:
- requirements.txt includes required packages
- no local absolute paths like /Users/bz/...
- model and data files are committed
- app works from repo root
- Streamlit Cloud can locate the app file
- no secrets or API keys are committed
