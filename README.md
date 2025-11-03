# feature-selection-lab
Feature Selection techniques on Breast Cancer Dataset
# Feature Selection Lab - Modified Version

**Student**: Uzma
**Assignment**: Feature Selection Techniques Analysis  
**Dataset**: Breast Cancer Wisconsin (Diagnostic)  

##  Modifications Made

This lab has been significantly enhanced with the following modifications:

### 1. **Data Loading Enhancement**
-  Uses sklearn's built-in breast cancer dataset instead of CSV loading
-  Better data reproducibility and standardization
-  Automatic feature naming and target encoding

### 2. **Evaluation Enhancement**
-  Added cross-validation for more robust performance estimation
-  Extended metrics including CV F1 mean and standard deviation
-  Better statistical evaluation of model performance

### 3. **Visualization Enhancements**
-  Interactive correlation matrix using Plotly
-  Feature importance visualization
-  F-test score visualization with color coding
-  Comprehensive comparison dashboard

### 4. **Analysis Features**
-  Automated feature correlation analysis
-  Feature ranking visualization for RFE
-  Side-by-side method comparison
-  Automated insights and recommendations

### 5. **Additional Features**
-  Results export to CSV
-  Professional documentation
-  Code quality improvements
-  Enhanced error handling

##  How to Run

1. **Install Dependencies**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

2. **Run the Notebook**:
```bash
jupyter notebook Feature_Selection_Modified.ipynb
```

##  Key Results

The analysis compares multiple feature selection techniques:
- **Filter Methods**: Correlation analysis, F-test
- **Wrapper Methods**: Recursive Feature Elimination (RFE)
- **Embedded Methods**: Random Forest Feature Importance, L1 Regularization

##  Key Findings

- All feature selection methods achieve similar high performance
- F-test provides good balance between performance and efficiency
- Cross-validation confirms robust performance across different data splits
- Interactive visualizations provide better insights into feature relationships

##  Repository Structure

```
feature-selection-lab/
├── Feature_Selection_Modified.ipynb    # Main modified notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── feature_selection_results.csv     # Results output (generated)
```

##  Learning Outcomes

This modified lab demonstrates:
1. Advanced feature selection techniques
2. Cross-validation for robust evaluation
3. Interactive data visualization
4. Comparative analysis methods
5. Professional code documentation

##  GitHub Repository

Repository URL: `https://github.com/uXmii/feature-selection-lab`

---
*Modified version created for educational purposes as part of machine learning coursework.*