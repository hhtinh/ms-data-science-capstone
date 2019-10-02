import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn import cross_validation, decomposition, grid_search
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Home made One Hot Encoder
def convert_to_binary(df, column_to_convert):
    categories = list(df[column_to_convert].drop_duplicates())

    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("/", "_").replace("-", "").replace("'", "").replace("&", "_").replace(":", "_").replace(",", "").lower()
        col_name = column_to_convert[:14] + '_' + cat_name[:35]
        df[col_name] = 0
        df.loc[(df[column_to_convert] == category), col_name] = 1

    return df

# Import data
print("Reading in data...")
df_train = pd.read_csv("Training_set_values.csv", header=0, index_col=None)

df_label = pd.read_csv("Training_set_labels.csv", header=0, index_col=None)
df_test = pd.read_csv("Test_set_values.csv", header=0, index_col=None)

# Combine into one dataset
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# Imputing numeric columns
print("Imputing numeric columns...")
columns_to_fill_mean = ['admissions__act_scores_25th_percentile_cumulative',\
'admissions__act_scores_25th_percentile_english',\
'admissions__act_scores_25th_percentile_math',\
'admissions__act_scores_25th_percentile_writing',\
'admissions__act_scores_75th_percentile_cumulative',\
'admissions__act_scores_75th_percentile_english',\
'admissions__act_scores_75th_percentile_math',\
'admissions__act_scores_75th_percentile_writing',\
'admissions__act_scores_midpoint_cumulative',\
'admissions__act_scores_midpoint_english',\
'admissions__act_scores_midpoint_math',\
'admissions__act_scores_midpoint_writing',\
'admissions__sat_scores_25th_percentile_critical_reading',\
'admissions__sat_scores_25th_percentile_math',\
'admissions__sat_scores_25th_percentile_writing',\
'admissions__sat_scores_75th_percentile_critical_reading',\
'admissions__sat_scores_75th_percentile_math',\
'admissions__sat_scores_75th_percentile_writing',\
'admissions__sat_scores_average_by_ope_id',\
'admissions__sat_scores_average_overall',\
'admissions__sat_scores_midpoint_critical_reading',\
'admissions__sat_scores_midpoint_math',\
'admissions__sat_scores_midpoint_writing',\
'aid__cumulative_debt_10th_percentile',\
'aid__cumulative_debt_25th_percentile',\
'aid__cumulative_debt_75th_percentile',\
'aid__cumulative_debt_90th_percentile',\
'aid__cumulative_debt_number',\
'aid__loan_principal',\
'completion__completion_cohort_4yr_100nt',\
'completion__completion_cohort_4yr_150nt',\
'completion__completion_cohort_4yr_150nt_pooled',\
'completion__completion_cohort_less_than_4yr_100nt',\
'completion__completion_cohort_less_than_4yr_150nt',\
'completion__completion_cohort_less_than_4yr_150nt_pooled',\
'cost__attendance_academic_year',\
'cost__attendance_program_year',\
'cost__avg_net_price_private',\
'cost__avg_net_price_public',\
'cost__net_price_private_by_income_level_0_30000',\
'cost__net_price_private_by_income_level_0_48000',\
'cost__net_price_private_by_income_level_110001_plus',\
'cost__net_price_private_by_income_level_30001_48000',\
'cost__net_price_private_by_income_level_30001_75000',\
'cost__net_price_private_by_income_level_48001_75000',\
'cost__net_price_private_by_income_level_75000_plus',\
'cost__net_price_private_by_income_level_75001_110000',\
'cost__net_price_public_by_income_level_0_30000',\
'cost__net_price_public_by_income_level_0_48000',\
'cost__net_price_public_by_income_level_110001_plus',\
'cost__net_price_public_by_income_level_30001_48000',\
'cost__net_price_public_by_income_level_30001_75000',\
'cost__net_price_public_by_income_level_48001_75000',\
'cost__net_price_public_by_income_level_75000_plus',\
'cost__net_price_public_by_income_level_75001_110000',\
'cost__title_iv_private_all',\
'cost__title_iv_private_by_income_level_0_30000',\
'cost__title_iv_private_by_income_level_110001_plus',\
'cost__title_iv_private_by_income_level_30001_48000',\
'cost__title_iv_private_by_income_level_48001_75000',\
'cost__title_iv_private_by_income_level_75001_110000',\
'cost__title_iv_public_all',\
'cost__title_iv_public_by_income_level_0_30000',\
'cost__title_iv_public_by_income_level_110001_plus',\
'cost__title_iv_public_by_income_level_30001_48000',\
'cost__title_iv_public_by_income_level_48001_75000',\
'cost__title_iv_public_by_income_level_75001_110000',\
'cost__tuition_in_state',\
'cost__tuition_out_of_state',\
'cost__tuition_program_year',\
'school__faculty_salary', 'student__valid_dependency_status']

for column in columns_to_fill_mean:
    df_all[column].fillna( df_all[column].mean(), inplace=True)

columns_to_fill_median = ['aid__median_debt_completers_monthly_payments',\
'aid__median_debt_completers_overall',\
'aid__median_debt_dependent_students',\
'aid__median_debt_female_students',\
'aid__median_debt_first_generation_students',\
'aid__median_debt_income_0_30000',\
'aid__median_debt_income_30001_75000',\
'aid__median_debt_income_greater_than_75000',\
'aid__median_debt_independent_students',\
'aid__median_debt_male_students',\
'aid__median_debt_no_pell_grant',\
'aid__median_debt_non_first_generation_students',\
'aid__median_debt_noncompleters',\
'aid__median_debt_number_completers',\
'aid__median_debt_number_dependent_students',\
'aid__median_debt_number_female_students',\
'aid__median_debt_number_first_generation_students',\
'aid__median_debt_number_income_0_30000',\
'aid__median_debt_number_income_30001_75000',\
'aid__median_debt_number_income_greater_than_75000',\
'aid__median_debt_number_independent_students',\
'aid__median_debt_number_male_students',\
'aid__median_debt_number_no_pell_grant',\
'aid__median_debt_number_non_first_generation_students',\
'aid__median_debt_number_noncompleters',\
'aid__median_debt_number_overall',\
'aid__median_debt_number_pell_grant',\
'aid__median_debt_pell_grant']

for column in columns_to_fill_median:
    df_all[column].fillna( df_all[column].median(), inplace=True)

# Imputing category columns
print("Imputing category columns...")
columns_to_convert = [#'report_year',\
'school__institutional_characteristics_level',\
'school__locale',\
'school__main_campus',\
'school__minority_serving_aanipi',\
'school__minority_serving_annh',\
'school__minority_serving_hispanic',\
'school__minority_serving_historically_black',\
'school__minority_serving_nant',\
'school__minority_serving_predominantly_black',\
'school__minority_serving_tribal',\
'school__ownership',\
'school__region_id',\
'school__religious_affiliation',\
'school__state']

for column in columns_to_convert:
    df_all[column].fillna("Unknown", inplace=True)

# One Hot Encoding
print("One Hot Encoding categorical data...")

for column in columns_to_convert:
    df_all = convert_to_binary(df=df_all, column_to_convert=column)
    df_all.drop(column, axis=1, inplace=True)

# Drop unnecessary columns
columns_to_drop = ['report_year',\
'school__men_only', 'school__women_only',
'school__online_only',\
'school__carnegie_basic',\
'school__carnegie_size_setting',\
'school__carnegie_undergrad',\
'school__degrees_awarded_highest',\
'school__degrees_awarded_predominant',\
'student__share_firstgeneration_parents_highschool',\
'academics__program_bachelors_agriculture',\
'academics__program_bachelors_architecture',\
'academics__program_bachelors_biological',\
'academics__program_bachelors_business_marketing',\
'academics__program_bachelors_communication',\
'academics__program_bachelors_communications_technology',\
'academics__program_bachelors_computer',\
'academics__program_bachelors_construction',\
'academics__program_bachelors_education',\
'academics__program_bachelors_engineering',\
'academics__program_bachelors_engineering_technology',\
'academics__program_bachelors_english',\
'academics__program_bachelors_ethnic_cultural_gender',\
'academics__program_bachelors_family_consumer_science',\
'academics__program_bachelors_health',\
'academics__program_bachelors_history',\
'academics__program_bachelors_humanities',\
'academics__program_bachelors_language',\
'academics__program_bachelors_legal',\
'academics__program_bachelors_library',\
'academics__program_bachelors_mathematics',\
'academics__program_bachelors_mechanic_repair_technology',\
'academics__program_bachelors_military',\
'academics__program_bachelors_multidiscipline',\
'academics__program_bachelors_parks_recreation_fitness',\
'academics__program_bachelors_personal_culinary',\
'academics__program_bachelors_philosophy_religious',\
'academics__program_bachelors_physical_science',\
'academics__program_bachelors_precision_production',\
'academics__program_bachelors_psychology',\
'academics__program_bachelors_public_administration_social_service',\
'academics__program_bachelors_resources',\
'academics__program_bachelors_science_technology',\
'academics__program_bachelors_security_law_enforcement',\
'academics__program_bachelors_social_science',\
'academics__program_bachelors_theology_religious_vocation',\
'academics__program_bachelors_transportation',\
'academics__program_bachelors_visual_performing',\
'academics__program_certificate_lt_1_yr_agriculture',\
'academics__program_certificate_lt_1_yr_architecture',\
'academics__program_certificate_lt_1_yr_biological',\
'academics__program_certificate_lt_1_yr_business_marketing',\
'academics__program_certificate_lt_1_yr_communication',\
'academics__program_certificate_lt_1_yr_communications_technology',\
'academics__program_certificate_lt_1_yr_computer',\
'academics__program_certificate_lt_1_yr_construction',\
'academics__program_certificate_lt_1_yr_education',\
'academics__program_certificate_lt_1_yr_engineering',\
'academics__program_certificate_lt_1_yr_engineering_technology',\
'academics__program_certificate_lt_1_yr_english',\
'academics__program_certificate_lt_1_yr_ethnic_cultural_gender',\
'academics__program_certificate_lt_1_yr_family_consumer_science',\
'academics__program_certificate_lt_1_yr_health',\
'academics__program_certificate_lt_1_yr_history',\
'academics__program_certificate_lt_1_yr_humanities',\
'academics__program_certificate_lt_1_yr_language',\
'academics__program_certificate_lt_1_yr_legal',\
'academics__program_certificate_lt_1_yr_library',\
'academics__program_certificate_lt_1_yr_mathematics',\
'academics__program_certificate_lt_1_yr_mechanic_repair_technology',\
'academics__program_certificate_lt_1_yr_military',\
'academics__program_certificate_lt_1_yr_multidiscipline',\
'academics__program_certificate_lt_1_yr_parks_recreation_fitness',\
'academics__program_certificate_lt_1_yr_personal_culinary',\
'academics__program_certificate_lt_1_yr_philosophy_religious',\
'academics__program_certificate_lt_1_yr_physical_science',\
'academics__program_certificate_lt_1_yr_precision_production',\
'academics__program_certificate_lt_1_yr_psychology',\
'academics__program_certificate_lt_1_yr_public_administration_social_service',\
'academics__program_certificate_lt_1_yr_resources',\
'academics__program_certificate_lt_1_yr_science_technology',\
'academics__program_certificate_lt_1_yr_security_law_enforcement',\
'academics__program_certificate_lt_1_yr_social_science',\
'academics__program_certificate_lt_1_yr_theology_religious_vocation',\
'academics__program_certificate_lt_1_yr_transportation',\
'academics__program_certificate_lt_1_yr_visual_performing',\
'academics__program_certificate_lt_2_yr_agriculture',\
'academics__program_certificate_lt_2_yr_architecture',\
'academics__program_certificate_lt_2_yr_biological',\
'academics__program_certificate_lt_2_yr_business_marketing',\
'academics__program_certificate_lt_2_yr_communication',\
'academics__program_certificate_lt_2_yr_communications_technology',\
'academics__program_certificate_lt_2_yr_computer',\
'academics__program_certificate_lt_2_yr_construction',\
'academics__program_certificate_lt_2_yr_education',\
'academics__program_certificate_lt_2_yr_engineering',\
'academics__program_certificate_lt_2_yr_engineering_technology',\
'academics__program_certificate_lt_2_yr_english',\
'academics__program_certificate_lt_2_yr_ethnic_cultural_gender',\
'academics__program_certificate_lt_2_yr_family_consumer_science',\
'academics__program_certificate_lt_2_yr_health',\
'academics__program_certificate_lt_2_yr_history',\
'academics__program_certificate_lt_2_yr_humanities',\
'academics__program_certificate_lt_2_yr_language',\
'academics__program_certificate_lt_2_yr_legal',\
'academics__program_certificate_lt_2_yr_library',\
'academics__program_certificate_lt_2_yr_mathematics',\
'academics__program_certificate_lt_2_yr_mechanic_repair_technology',\
'academics__program_certificate_lt_2_yr_military',\
'academics__program_certificate_lt_2_yr_multidiscipline',\
'academics__program_certificate_lt_2_yr_parks_recreation_fitness',\
'academics__program_certificate_lt_2_yr_personal_culinary',\
'academics__program_certificate_lt_2_yr_philosophy_religious',\
'academics__program_certificate_lt_2_yr_physical_science',\
'academics__program_certificate_lt_2_yr_precision_production',\
'academics__program_certificate_lt_2_yr_psychology',\
'academics__program_certificate_lt_2_yr_public_administration_social_service',\
'academics__program_certificate_lt_2_yr_resources',\
'academics__program_certificate_lt_2_yr_science_technology',\
'academics__program_certificate_lt_2_yr_security_law_enforcement',\
'academics__program_certificate_lt_2_yr_social_science',\
'academics__program_certificate_lt_2_yr_theology_religious_vocation',\
'academics__program_certificate_lt_2_yr_transportation',\
'academics__program_certificate_lt_2_yr_visual_performing',\
'academics__program_certificate_lt_4_yr_agriculture',\
'academics__program_certificate_lt_4_yr_architecture',\
'academics__program_certificate_lt_4_yr_biological',\
'academics__program_certificate_lt_4_yr_business_marketing',\
'academics__program_certificate_lt_4_yr_communication',\
'academics__program_certificate_lt_4_yr_communications_technology',\
'academics__program_certificate_lt_4_yr_computer',\
'academics__program_certificate_lt_4_yr_construction',\
'academics__program_certificate_lt_4_yr_education',\
'academics__program_certificate_lt_4_yr_engineering',\
'academics__program_certificate_lt_4_yr_engineering_technology',\
'academics__program_certificate_lt_4_yr_english',\
'academics__program_certificate_lt_4_yr_ethnic_cultural_gender',\
'academics__program_certificate_lt_4_yr_family_consumer_science',\
'academics__program_certificate_lt_4_yr_health',\
'academics__program_certificate_lt_4_yr_history',\
'academics__program_certificate_lt_4_yr_humanities',\
'academics__program_certificate_lt_4_yr_language',\
'academics__program_certificate_lt_4_yr_legal',\
'academics__program_certificate_lt_4_yr_library',\
'academics__program_certificate_lt_4_yr_mathematics',\
'academics__program_certificate_lt_4_yr_mechanic_repair_technology',\
'academics__program_certificate_lt_4_yr_military',\
'academics__program_certificate_lt_4_yr_multidiscipline',\
'academics__program_certificate_lt_4_yr_parks_recreation_fitness',\
'academics__program_certificate_lt_4_yr_personal_culinary',\
'academics__program_certificate_lt_4_yr_philosophy_religious',\
'academics__program_certificate_lt_4_yr_physical_science',\
'academics__program_certificate_lt_4_yr_precision_production',\
'academics__program_certificate_lt_4_yr_psychology',\
'academics__program_certificate_lt_4_yr_public_administration_social_service',\
'academics__program_certificate_lt_4_yr_resources',\
'academics__program_certificate_lt_4_yr_science_technology',\
'academics__program_certificate_lt_4_yr_security_law_enforcement',\
'academics__program_certificate_lt_4_yr_social_science',\
'academics__program_certificate_lt_4_yr_theology_religious_vocation',\
'academics__program_certificate_lt_4_yr_transportation',\
'academics__program_certificate_lt_4_yr_visual_performing',\
'academics__program_assoc_agriculture',\
'academics__program_assoc_architecture',\
'academics__program_assoc_biological',\
'academics__program_assoc_business_marketing',\
'academics__program_assoc_communication',\
'academics__program_assoc_communications_technology',\
'academics__program_assoc_computer',\
'academics__program_assoc_construction',\
'academics__program_assoc_education',\
'academics__program_assoc_engineering',\
'academics__program_assoc_engineering_technology',\
'academics__program_assoc_english',\
'academics__program_assoc_ethnic_cultural_gender',\
'academics__program_assoc_family_consumer_science',\
'academics__program_assoc_health',\
'academics__program_assoc_history',\
'academics__program_assoc_humanities',\
'academics__program_assoc_language',\
'academics__program_assoc_legal',\
'academics__program_assoc_library',\
'academics__program_assoc_mathematics',\
'academics__program_assoc_mechanic_repair_technology',\
'academics__program_assoc_military',\
'academics__program_assoc_multidiscipline',\
'academics__program_assoc_parks_recreation_fitness',\
'academics__program_assoc_personal_culinary',\
'academics__program_assoc_philosophy_religious',\
'academics__program_assoc_physical_science',\
'academics__program_assoc_precision_production',\
'academics__program_assoc_psychology',\
'academics__program_assoc_public_administration_social_service',\
'academics__program_assoc_resources',\
'academics__program_assoc_science_technology',\
'academics__program_assoc_security_law_enforcement',\
'academics__program_assoc_social_science',\
'academics__program_assoc_theology_religious_vocation',\
'academics__program_assoc_transportation',\
'academics__program_assoc_visual_performing']

for column in columns_to_drop:
    if column in df_all.columns:
        df_all.drop(column, axis=1, inplace=True)

# Preparing datasets for training model
print("Preparing datasets for training model...")

df_all.set_index('row_id', inplace=True)
df_label.set_index('row_id', inplace=True)
df_test.set_index('row_id', inplace=True)

df_train = pd.concat([df_label['repayment_rate'], df_all], axis=1, join='inner')
# Drop too many NaNs rows
#df_train = df_train.dropna(axis=0, thresh=280)

y = df_train['repayment_rate']
X = df_train.drop('repayment_rate', axis=1)

X = X.fillna(0)

# Cross validating and param grid searching
print("Cross validating and param grid searching...")

param_grid = {'n_estimators': [1600]}
LGB_model = lgb.LGBMRegressor(objective='regression', boosting_type='gbdt', metric='l2_root',\
max_depth=6, subsample=0.9, colsample_bytree=0.8, learning_rate=0.04, seed=0)
model = grid_search.GridSearchCV(estimator=LGB_model, param_grid=param_grid,\
scoring='neg_mean_squared_error', verbose=10, n_jobs=3, iid=True, refit=True, cv=5)

model.fit(X, y)

print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

"""
seed = 8
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Training model
print("Training model...")

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
 'task': 'train',
 'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse', #{'l2_root', 'l2'},
 'num_leaves': 120,
 'max_depth': 6,
 'learning_rate': 0.1,
 'min_data_in_leaf': 500,
 'feature_fraction': 0.9,
 'bagging_fraction': 0.9,
 'bagging_freq': 10,
 'verbose': 0
}

params = {
 'task': 'train',
 'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse',
 'num_leaves': 33,
 'max_depth': 6,
 'learning_rate': 0.0395,
 'min_child_samples': 11,
 'feature_fraction': 0.9,
 'verbose': 5
}

print('Start training...')
# train
gbm = lgb.train(params,
 lgb_train,
 num_boost_round=500,
 valid_sets=lgb_eval,
 early_stopping_rounds=10)

# predict
y_pred = gbm.predict(X_test, num_iteration = gbm.best_iteration)
# evaluate
print('The RMSE of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# Feature importance
print('Feature importance results:')

df = pd.DataFrame(np.column_stack((gbm.feature_name(), gbm.feature_importance())), columns=['name', 'importance'])
df.importance = pd.to_numeric(df.importance, errors='coerce')
print(df.sort_values(by='importance', ascending=False).head(50))
"""
# Preparing data for submission
print("Preparing data for submission...")

df_test = pd.merge(df_test.loc[:,['report_year']], df_all, how='left', left_index=True, right_index=True, sort=False)
X_submit = df_test.drop('report_year', axis=1, inplace=False)
id_submit = df_test.index.values

X_submit = X_submit.fillna(0)

#y_no_cv = gbm.predict(X_submit, num_iteration=gbm.best_iteration)
y_cv = model.predict(X_submit)

#sub_no_cv = pd.DataFrame(np.column_stack((id_submit, y_no_cv)), columns=['row_id', 'repayment_rate'])
#sub_no_cv.to_csv('submission_no_cv.csv', index=False)

sub_cv = pd.DataFrame(np.column_stack((id_submit, y_cv)), columns=['row_id', 'repayment_rate'])
sub_cv.to_csv('submission_lgb-98-040-dropNA.csv', index=False)
