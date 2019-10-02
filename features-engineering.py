
# Features engineering
print("Features engineering...")

df_all['student__family_income_dependent_rate'] = df_all.student__family_income_dependent_students / df_all.student__family_income_overall
df_all['student__family_income_dependent_rate'].fillna( df_all['student__family_income_dependent_rate'].mean(), inplace=True)

df_all['student__family_income_independent_rate'] = df_all.student__family_income_independent_students / df_all.student__family_income_overall
df_all['student__family_income_independent_rate'].fillna( df_all['student__family_income_independent_rate'].mean(), inplace=True)

df_all['student__family_income_dependent_independent_rate_diff'] = df_all.student__family_income_independent_rate - df_all.student__family_income_dependent_rate

df_all['cost__net_price_private_by_income_level_0_75000'] = df_all.cost__net_price_private_by_income_level_0_48000 + df_all.cost__net_price_private_by_income_level_48001_75000
df_all['cost__net_price_private_by_income_level_0_110000'] = df_all.cost__net_price_private_by_income_level_0_75000 + df_all.cost__net_price_private_by_income_level_75001_110000

df_all['cost__net_price_public_by_income_level_0_75000'] = df_all.cost__net_price_public_by_income_level_0_48000 + df_all.cost__net_price_public_by_income_level_48001_75000
df_all['cost__net_price_public_by_income_level_0_110000'] = df_all.cost__net_price_public_by_income_level_0_75000 + df_all.cost__net_price_public_by_income_level_75001_110000

df_all['cost__title_iv_private_by_income_level_0_48000'] = df_all.cost__title_iv_private_by_income_level_0_30000 + df_all.cost__title_iv_private_by_income_level_30001_48000
df_all['cost__title_iv_private_by_income_level_0_75000'] = df_all.cost__title_iv_private_by_income_level_0_48000 + df_all.cost__title_iv_private_by_income_level_48001_75000
df_all['cost__title_iv_private_by_income_level_0_110000'] = df_all.cost__title_iv_private_by_income_level_0_75000 + df_all.cost__title_iv_private_by_income_level_75001_110000

df_all['cost__title_iv_public_by_income_level_0_48000'] = df_all.cost__title_iv_public_by_income_level_0_30000 + df_all.cost__title_iv_public_by_income_level_30001_48000
df_all['cost__title_iv_public_by_income_level_0_75000'] = df_all.cost__title_iv_public_by_income_level_0_48000 + df_all.cost__title_iv_public_by_income_level_48001_75000
df_all['cost__title_iv_public_by_income_level_0_110000'] = df_all.cost__title_iv_public_by_income_level_0_75000 + df_all.cost__title_iv_public_by_income_level_75001_110000
