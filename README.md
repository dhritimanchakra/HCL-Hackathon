# House Price Prediction API (HCLTech Hackathon)

## Problem Statement
Predicting real estate prices is crucial for buyers, sellers, and investors. This project aims to build a robust and highly accurate predictive model for house prices using historical and structural data. By providing an accessible API, we enable seamless integration into web and mobile applications for real-time property valuation.

## High-Level Architecture
Our system is designed for scalability and real-time inference:

1.  **Data Ingestion & Cleaning:** Raw housing datasets are loaded and cleaned (handling missing values, outliers).
2.  **Preprocessing & Feature Engineering:** Domain-specific transformations, encoding categorical variables, and scaling numerical features.
3.  **XGBoost Model:** A gradient boosting model trained to capture complex non-linear relationships in housing data.
4.  **FastAPI Deployment:** The trained model is served via a high-performance REST API built with FastAPI and connected to a MongoDB backend for logging and analytics.

## Domain Adaptations (India/Temporal Context)
To make this model relevant for the specific hackathon problem space:

*   **Zoning to Tier Mapping:** Traditional (US-centric) zoning classifications are mapped and adapted to Indian Urban/Rural tiers (e.g., mapping specific zones to characteristics similar to Tier 1 cities like Chennai or Tier 2 cities like Vellore).
*   **Temporal Appreciation Simulation:** The model incorporates logic to simulate and project real estate appreciation specifically targeting the 2026-2028 timeframe, accounting for anticipated market trends and inflation.
