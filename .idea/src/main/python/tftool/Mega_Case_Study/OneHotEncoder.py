import pandas as pd
from sklearn.preprocessing import OneHotEncoder

colors_df = pd.DataFrame(data=[[5,'yellow'],[1,'red'],[2,'blue'],[3,'green'],[4,'blue'],[7,'purple']], columns=['id', 'color'])
print('Before One Hot Encoding:')
display(colors_df)

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(colors_df)

colors_df_encoded = one_hot_encoder.transform(colors_df)
colors_df_encoded = pd.DataFrame(data=colors_df_encoded, columns=one_hot_encoder.categories_)
print('\nAfter One Hot Encoding:')
display(colors_df_encoded)