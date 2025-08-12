# Campus Care

Campus Care is a student-centered Streamlit app that helps university students reflect on their wellbeing, get nudged toward the right support, and track progress over time. It combines a short self-assessment, a risk-aware pathway with personalized insights, and a dashboard with badges and reflections — all backed by a MySQL database and a lightweight ML model.

- Live UX highlights:
  - Guided 7-question SWEMWBS self-check
  - Smart branching to Low-Risk or High-Risk pathways
  - Cluster assignment and radar insights for higher-risk users
  - Micro-modules with journaling and persistence
  - Progress dashboard with badges and recent cluster visits

---

## Features

- Authentication
  - Sign up and log in with bcrypt-hashed passwords
- Self Assessment (SWEMWBS-inspired)
  - 7 sliders (1–5) compute a total score to determine pathway
- Pathways
  - Low-Risk: short awareness modules, reflections, and habit nudges
  - High-Risk: deeper intake (sleep, study, stress, suicidal ideation, etc.), ML scoring, cluster insights
- Personalized Insights
  - Moderate/Severe users get assigned to cluster profiles drawn from a curated CSV
  - Radar chart comparing profile features
- Dashboard
  - Summary of check-ins (total / low / high)
  - Recent high-risk cluster visits with descriptions
  - Achievement-style badges based on behavior
  - Latest module reflections

---
