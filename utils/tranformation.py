import pandas as pd


def format_input_to_dict(input_dict):
    numeric_features = ['avg_storey_range', 'floor_area_sqm', 'total_dwelling_units', 'vacancy',
                        'commercial', 'mrt_interchange', 'age_of_flat']

    all_features = ['avg_storey_range', 'floor_area_sqm', 'total_dwelling_units', 'vacancy',
                    'commercial', 'mrt_interchange', 'age_of_flat', 'flat_model_model a',
                    'flat_model_new generation', 'flat_model_other',
                    'flat_model_premium apartment', 'town_jurong west', 'town_other',
                    'town_punggol', 'town_sengkang', 'town_tampines', 'town_woodlands',
                    'town_yishun']

    scale_dic = {'avg_storey_range': (2.0, 50.0),
                 'floor_area_sqm': (31.0, 280.0),
                 'total_dwelling_units': (2, 570),
                 'vacancy': (20, 110)}

    def scale_down_datapoint(datapoint, min_values, max_values):
        scaled_datapoint = (datapoint - min_values) / (max_values - min_values)
        return scaled_datapoint

    # initialise results dictionary, setting all features to 0
    results = {feature: 0 for feature in all_features}

    for key, value in input_dict.items():
        if key in numeric_features:
            scaled_value = value
            if key in scale_dic:
                scaled_value = scale_down_datapoint(value, scale_dic[key][0],
                                                    scale_dic[key][1])
            results[key] = scaled_value
        else:
            formatted_key = ''
            # handle categorical features
            if key == 'flat_model':
                formatted_key = 'flat_model_' + value.lower()
            elif key == 'town':
                formatted_key = 'town_' + value.lower()
            print(formatted_key)

            if formatted_key in results:
                results[formatted_key] = 1

    return results


def lr_prediction(model, input_dict):
    input_dict = pd.DataFrame([input_dict])
    filtered_features = ['floor_area_sqm', 'avg_storey_range', 'total_dwelling_units',
                         'commercial', 'flat_model_model a', 'flat_model_new generation',
                         'flat_model_other', 'flat_model_premium apartment']
    results = model.predict(input_dict[filtered_features])

    min_price, max_price = (154209.1203140869, 1540198.2919407086)
    rescaled_datapoint = sum(results) * (max_price - min_price) + min_price
    return rescaled_datapoint
