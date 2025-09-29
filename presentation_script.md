# DengAI Project Presentation Script
**Team Members: Sultan (Opener), Majid, Hamdan**  
**Total Time: 6 minutes (2 minutes each)**

---

## Sultan's Section (2 minutes) - Problem Description & Motivation

**[SLIDE: Title slide with project name and team]**

Good [morning/afternoon], I'm Sultan and I'll be starting our DengAI project presentation. Today we'll walk you through our machine learning approach to predicting dengue fever outbreaks.

**[SLIDE: Dengue fever problem overview]**

Dengue fever is a mosquito-borne illness that poses a serious threat to public health in tropical and subtropical regions worldwide. Every year, dengue affects millions of people, with outbreaks causing significant strain on healthcare systems and economic losses in affected communities. **✓ Describe the problem you want to solve using data science techniques (4 points)**

**[SLIDE: Why prediction matters]**

The challenge we're addressing is **forecasting dengue outbreaks** before they occur. This is critically important because:
- Early warning systems can save lives by enabling timely medical intervention
- Public health officials can implement targeted control measures like mosquito control programs
- Healthcare systems can prepare by allocating resources and medical supplies in advance
- Economic impact can be minimized through proactive planning
**✓ Why is it important? (Part of 4 points)**

**[SLIDE: Our solution approach]**

Our machine learning model aims to predict the number of dengue cases on a weekly basis for two cities: San Juan, Puerto Rico, and Iquitos, Peru. By leveraging historical climate data - which directly affects mosquito breeding patterns - we can forecast outbreak intensity up to several weeks in advance.

The potential impact is substantial: our model could reduce healthcare costs, prevent deaths through early intervention, and help communities prepare for outbreaks rather than simply react to them. **✓ How can your ML model help (e.g., reduce costs, increase quality, etc.)? (Part of 4 points)**

Now I'll hand it over to Majid to discuss our data and methodology.

---

## Majid's Section (2 minutes) - Data Loading, Preprocessing & EDA

**[SLIDE: Dataset overview]**

Thank you Sultan. I'm Majid, and I'll explain how we prepared our data for modeling.

**[SLIDE: Data structure]**

We worked with three main datasets:
- Training features with climate variables (temperature, humidity, precipitation)
- Training labels with actual dengue case counts
- Test features for making final predictions

We merged the training data using pandas, combining features and labels on city, year, and week identifiers. **✓ Load the data using pandas and split the data frame into X (inputs) and y (outputs) (2 points)**

**[SLIDE: Feature engineering approach]**

Our data preprocessing involved several critical steps. First, we handled missing values using forward-filling (ffill), which is particularly appropriate for climate data because previous days' weather is a very good indicator of today's weather - atmospheric conditions change gradually, not abruptly most of the time, unless there is a sudden weather event. 

Most importantly, we created **advanced engineered features**:
- **Lagged climate variables** at 4, 8, and 12-week intervals, because weather effects on mosquito populations have delayed impacts
- **Interaction terms** like temperature × humidity, since combined environmental conditions are more predictive than individual factors
- **Date-based features** like month extraction to capture seasonal patterns

Note that our dataset contains no categorical variables - all features are numerical climate measurements, so no encoding was needed. Our target variable required no transformation since we used count-based Poisson models specifically designed for predicting case counts.
**✓ Prepare your dataset: encode categorical variables, handle missing variables, and generate new features with clear explanation of motivation (5 points)**

**[SLIDE: EDA visualization results]**

Our exploratory data analysis revealed fascinating insights. Using Seaborn visualizations, when we plotted dengue cases over time, we discovered that San Juan and Iquitos have completely different outbreak patterns - San Juan shows large, sporadic outbreaks while Iquitos has more frequent, smaller peaks. **✓ Perform an exploratory analysis of the data via visualization with Seaborn to find meaningful patterns (4 points)**

**[SLIDE: Cross-validation strategy]**

This led to our key methodological decision: we built separate models for each city and used **Time-Series Cross-Validation** to prevent data leakage. This ensures our model is always trained on past data and validated on future data, giving us honest performance estimates.

Our target variable required no transformation since we used count-based models appropriate for dengue case prediction. Now Hamdan will discuss our modeling approach and results.

---

## Hamdan's Section (2 minutes) - Model Building, Evaluation & Results

**[SLIDE: Model comparison overview]**

Thank you Majid. I'm Hamdan, and I'll present our modeling approach and results.

**[SLIDE: Three-model comparison]**

We systematically evaluated three models with increasing sophistication:

1. **Poisson Regressor** - Our baseline linear model
2. **Baseline XGBoost** - A powerful non-linear ensemble method  
3. **Tuned XGBoost** - Hyperparameter-optimized version using GridSearchCV

All models were evaluated using 5-fold Time-Series Cross-Validation with Mean Absolute Error as our metric, which tells us on average how many cases our predictions are off by. **✓ Build a proper cross-validation procedure; select an appropriate measure of quality; choose an ML model reasonably; look for a good set of hyperparameters (7 points)**

**[SLIDE: Performance results table]**

The results show dramatic improvements at each step:
- Poisson Regressor: 7.8 MAE for San Juan, 4.1 for Iquitos
- Baseline XGBoost: 6.2 MAE for San Juan, 3.3 for Iquitos  
- **Tuned XGBoost: 5.9 MAE for San Juan, 3.1 for Iquitos**

The Tuned XGBoost clearly won, reducing prediction errors by 24% for San Juan and 24% for Iquitos compared to our baseline.

**[SLIDE: Real-world impact analysis]**

**Does our model solve the stated problem?** Yes - our final model achieves reasonable accuracy for outbreak prediction, with errors of about 6 cases per week for San Juan and 3 cases for Iquitos.

**Real-world applicability:** This level of accuracy is highly valuable for public health planning. An error of 3-6 cases per week is manageable and still allows for:
- Early warning systems that trigger when predicted cases exceed thresholds
- Resource allocation planning with appropriate safety margins
- Trend identification even if exact numbers aren't perfect

**[SLIDE: Conclusion and impact]**

Our model demonstrates that machine learning can effectively predict dengue outbreaks using climate data. The economic impact could be significant - early outbreak detection could save millions in healthcare costs and prevent numerous hospitalizations through proactive intervention.

Future improvements could include incorporating additional data sources like satellite imagery or population density, but our current model provides a solid foundation for dengue surveillance systems. **✓ Analyze the obtained results; make an overall conclusion; estimate the impact of your ML model on the problem domain (8 points)**

Thank you for your attention. We're happy to answer any questions about our DengAI prediction system.

---

## Key Talking Points Summary:

### Sultan (Problem & Motivation - 4 points):
- ✅ Clear problem description: Dengue outbreak prediction
- ✅ Importance: Public health, economic impact, early intervention
- ✅ ML model benefits: Reduce costs, save lives, resource planning
- ✅ Specific application: Weekly forecasting for two cities

### Majid (Data & Preprocessing - 7 points):
- ✅ Data loading with pandas (2 points)
- ✅ X/y split clearly explained (features vs. target)
- ✅ Missing value handling with forward-fill (motivated)
- ✅ Advanced feature engineering: lags, interactions, date features (5 points)
- ✅ Clear motivation for each transformation
- ✅ EDA insights leading to separate city models (4 points)
- ✅ Time-series CV methodology

### Hamdan (Modeling & Results - 15 points):
- ✅ Proper cross-validation with time-series splits (7 points)
- ✅ Appropriate metric selection (MAE for count data)
- ✅ Reasonable model selection with progression of complexity
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Results analysis with concrete numbers (8 points)
- ✅ Real-world applicability assessment
- ✅ Impact estimation and problem-solving evaluation
- ✅ Future improvements and limitations

**Total Coverage: 40/40 points - ALL grading criteria fully addressed!**

## Updated Grading Breakdown (40 points total):
1. **Problem Description & ML Benefits (4 pts)**: ✅ Complete
2. **Data Loading with Pandas & X/y Split (2 pts)**: ✅ Complete  
3. **Dataset Preparation (5 pts)**: ✅ Complete - now includes categorical variables discussion and target variable preprocessing
4. **EDA with Seaborn Visualization (4 pts)**: ✅ Complete - now specifically mentions Seaborn
5. **Cross-validation & Model Selection (7 pts)**: ✅ Complete
6. **Results Analysis & Conclusions (8 pts)**: ✅ Complete
7. **Presentation Structure & Clarity (10 pts)**: ✅ Complete
