# Survival Analysis Methods Review

## 1. Literature Review: GA-GCN Paper by Ramirez (2022)

**Investigation Results:**
An exhaustive literature search for the exact paper titled *"Survival analysis using a graph-augmented graph convolution neural network (GA-GCN) for breast cancer relapse prediction"* (Ramirez 2022) yields **no matching results**. It appears this citation is a hallucination or an amalgamation of different papers:

- **Closest Match 1:** *Prediction and interpretation of cancer survival using graph convolution neural networks* by Ramirez et al., published in 2021 (often referred to as Surv_GCNN). 
- **Closest Match 2:** *Survival analysis using a graph-augmented graph convolution neural network* by Aghaeifar & Khasawneh (2026). 

Since the explicitly requested paper does not exist, the requested methodological details (GA-GCN node/edge types, Age/ER variables, and exact F1 scores for GSE2034, GSE2990, and BRCA) cannot be retrieved. *Caution is advised against using these specific fictitious metrics or paper details.*

## 2. Interpretable Survival Rule Learning Methods

### LR-Rules / RuleKit
- **Overview:** LR-Rules is a sequential covering (separate-and-conquer) rule induction algorithm designed for survival data.
- **Handling Right-Censored Data:** It handles right-censored outcomes natively in its rule-growing phase by using the **log-rank test** as the primary heuristic for evaluating the quality of the rule body ("IF" conditions).
- **Software Availability:** Implemented as part of the **RuleKit** suite. It provides a Java backend with a Python wrapper (scikit-learn compatible) and a web GUI (RuleMiner). Open-source on GitHub.

### GuideR
- **Overview:** A user-guided, separate-and-conquer rule learning algorithm introduced by Sikora, Wróbel, and Gudyś (2019). It operates across classification, regression, and survival analysis.
- **Handling Right-Censored Data:** It is specifically adapted to right-censored time-to-event data. Unlike fully automated rule learners, GuideR allows domain experts to impose constraints, preferences, and clinical knowledge to ensure generated rules are biologically or clinically relevant.
- **Software Availability:** Also integrated into the **RuleKit** / RuleMiner ecosystem.

### Survival Action Rules
- **Overview:** An extension of traditional action rules. Survival Action Rules suggest actionable interventions to improve patient prognosis.
- **Handling Right-Censored Data:** They operate natively on censored data by expressing conclusions as shifts in survival functions (e.g., Kaplan-Meier curves). The rule format is: IF [Changes in attributes] THEN [Change in expected survival curve].
- **Software Availability:** Predominantly custom research implementations (often detailed in MDPI publications). No major maintained open-source Python library exists identically named "Survival Action Rules"; they typically require bespoke implementation.

## 3. The "SWWFT" Method

**Investigation Results:**
A search for "SWWFT" within the context of survival analysis and interpretable ML yielded **no valid academic or software results**. It is not a real method. 

*Possibilities:*
- **Typo/Confusion:** It might be a typo for another acronym, or confused with **SurvSHAP(t)** / **SurvLIME**, which are the actual standard model-agnostic explanation tools for survival models. 
- **Hallucinated Acronym:** It may be a completely hallucinated term. 

Do not use "SWWFT" as a valid reference in your breast cancer survival pipeline.

