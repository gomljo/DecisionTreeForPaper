import pandas as pd

def label_processing(data_, num_of_classes=7):
    data = data_.copy()
    if num_of_classes == 2:
        data['NObeyesdad'] = data['NObeyesdad'].replace('Insufficient_Weight', 'Non-obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Normal_Weight', 'Non-obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Overweight_Level_I', 'Non-obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Overweight_Level_II', 'Non-obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Obesity_Type_I', 'obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Obesity_Type_II', 'obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Obesity_Type_III', 'obesity')
        after_labels = set(data['NObeyesdad'].tolist())
        print(after_labels)
        print('Whole labels: {}'.format(after_labels))
        print(data)

    if num_of_classes == 3:

        data['NObeyesdad'] = data['NObeyesdad'].replace('Insufficient_Weight', 'Non-obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Normal_Weight', 'Non-obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Overweight_Level_I', 'overweight')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Overweight_Level_II', 'overweight')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Obesity_Type_I', 'obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Obesity_Type_II', 'obesity')
        data['NObeyesdad'] = data['NObeyesdad'].replace('Obesity_Type_III', 'obesity')
        after_labels = set(data['NObeyesdad'].tolist())
        print(after_labels)
        print('Whole labels: {}'.format(after_labels))
        print(data)

    if num_of_classes == 7:
        after_labels = set(data['NObeyesdad'].tolist())
        print('Whole labels: {}'.format(after_labels))

    return data


def feature_processing(data):

    un_used_feature = ['Age', 'Height','Weight']

    data = data.drop(un_used_feature, axis=1)

    return data


def convert_string_to_numeric(data):

    col_lists = (data.columns)
    # vals = []
    for col in col_lists:

        vals = list(set(data[col].values))

        for i in range(len(vals)):
            data[col] = data[col].replace(vals[i], i)
    return data


