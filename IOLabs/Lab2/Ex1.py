import pandas as pd

# data_frame = pd.read_csv("iris.csv")
corrupted_data_frame = pd.read_csv("iris_with_errors.csv")

# print(data_frame)
# print(data_frame.values)
# print(data_frame.values[:, 0])
# print(data_frame.values[5:11, :])
# print(data_frame.values[1, 4])

# print(corrupted_data_frame.values)

# print(corrupted_data_frame.columns.size)
# print(corrupted_data_frame.index.size)

# Finding and displaying all errors in corrupted data frame
errors_array = []

for value in range(0, corrupted_data_frame.index.size):
     if (corrupted_data_frame.values[value, 4] != "Setosa" and corrupted_data_frame.values[value, 4] != "Versicolor"
             and corrupted_data_frame.values[value, 4] != "Virginica"):
         errors_array.append(corrupted_data_frame.values[value, :])

for column in range(0, corrupted_data_frame.columns.size - 1):
    for length in range(0, corrupted_data_frame.index.size):
        try:
            if corrupted_data_frame.values[length, column] == '-':
                errors_array.append(corrupted_data_frame.values[length, :])
            elif pd.isna(corrupted_data_frame.values[length, column]):
                errors_array.append(corrupted_data_frame.values[length, :])
            elif float(corrupted_data_frame.values[length, column]) < 0.0:
                errors_array.append(corrupted_data_frame.values[length, :])
            elif float(corrupted_data_frame.values[length, column]) > 15.0:
                errors_array.append(corrupted_data_frame.values[length, :])
        except ValueError:
            errors_array.append(corrupted_data_frame.values[length, :])
            pass

# print(float(corrupted_data_frame.values[0, 0]))

errors_data_frame = pd.DataFrame(errors_array, columns=corrupted_data_frame.columns)
errors_data_frame = errors_data_frame.drop_duplicates()
print(f"Number of errors is: {errors_data_frame.index.size + 1}")
print(errors_data_frame)

def average_of_row(data_frame: pd.core.frame.DataFrame, index, data_frame_column):
    row_values = []
    try:
        for size in range(0, data_frame.values[index, :].size - 1):
            if pd.isna(data_frame.values[index, size]):
                row_values.append(0)
            else:
                row_values.append(float(data_frame.values[index, size]))
    except ValueError:
        row_values.append(0)

    average_value = round(sum(row_values) / len(row_values), 1)
    data_frame.iloc[index, data_frame_column] = average_value

def correct_flower_name(data_frame: pd.core.frame.DataFrame, index):
    first_char = str(data_frame.values[index, data_frame.columns.size - 1])[0]
    second_char = str(data_frame.values[index, data_frame.columns.size - 1])[1]
    if first_char == 's':
        data_frame.iloc[index, data_frame.columns.size - 1] = "Setosa"
    elif first_char == 'v' or 'V':
        if second_char == 'e':
            data_frame.iloc[index, data_frame.columns.size - 1] = "Versicolor"
        else:
            data_frame.iloc[index, data_frame.columns.size - 1] = "Virginica"


# average_of_row(errors_data_frame, 0, 1)
# print(errors_data_frame.values[0, :])

# correct_flower_name(errors_data_frame, 0)
# print(errors_data_frame.values[0, :])

# Creation of a revised version of a data frame
print("The new, corrected data frame is shown below:")
corrected_data_frame = errors_data_frame
for value in range(0, corrected_data_frame.index.size):
     if (corrected_data_frame.values[value, 4] != "Setosa" and corrected_data_frame.values[value, 4] != "Versicolor"
             and corrected_data_frame.values[value, 4] != "Virginica"):
         correct_flower_name(corrected_data_frame, value)

     for column_num in range(0, corrected_data_frame.columns.size - 1):
         if corrected_data_frame.values[value, column_num] == '-':
             average_of_row(corrected_data_frame, value, column_num)
         elif pd.isna(corrected_data_frame.values[value, column_num]):
             average_of_row(corrected_data_frame, value, column_num)
         elif float(corrected_data_frame.values[value, column_num]) < 0.0:
             average_of_row(corrected_data_frame, value, column_num)
         elif float(corrected_data_frame.values[value, column_num]) > 15.0:
             average_of_row(corrected_data_frame, value, column_num)

print(corrected_data_frame)
