

### Reading files
1. Reading a large file in chunks
offers a crucial advantage when dealing with large datasets that may exceed available memory.
* Why use chunksize
    * Iterative data cleaning and transformation:
    You can clean or transform data in chunks, applying specific operations to each portion before combining them into a final, cleaned dataset.
    * Aggregating data from large files:
    For tasks like calculating sums, averages, or counts across a large dataset, you can aggregate within each chunk and then combine the aggregated results.

```
chunk_reader = pd.read_csv('data.csv', chunksize=100)

total_sum = 0
for i, chunk in enumerate(chunk_reader):
  total_sum += chunk['col1'].sum()


```

* Pandas agg allows
    * A simple string/callable : df.agg('sum')

    * Dictionary for column-specific aggregations (when we want to aggregate multiple columns at once) :  df.agg({'col1': 'sum', 'col2': ['mean', 'max']})

    * Named aggregation : df.agg(new_col_sum=('col1', 'sum'), new_col_mean=('col2', 'mean'))

* Group wise mean imputation using Pandas : In the following example we fill missing salary using average salary of the department
```

import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    "employee_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "department": ["HR", "HR", "IT", "IT", "Finance", "Finance", "Finance", "HR"],
    "salary": [50000, np.nan, 70000, np.nan, 60000, np.nan, 62000, np.nan],
    "experience_years": [2, 3, np.nan, 5, np.nan, 4, np.nan, np.nan]
}

df = pd.DataFrame(data)

mean_salary = df.groupby('department')['salary'].mean()
# print(mean_salary["HR"])
df['salary'] = df.apply(lambda x: mean_salary[x['department']] if pd.isnull(x['salary']) else x['salary'], axis=1)

# axis=1: This argument is essential for applying the function row-wise, allowing access to multiple columns within each row.
```

* Finding distance between consequtive pairs of rows
