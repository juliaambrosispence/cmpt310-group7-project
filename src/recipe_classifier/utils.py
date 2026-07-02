# utils.py
# Helper functions

# Function to count the frequency of elements in a string array column like ingredients or keyword
def count_string_frequency(elements_list, count):
    ingredients_dict = {}
    for ingredients in elements_list:
        for ingredient in ingredients:
            if ingredient is not None:
                if ingredient not in ingredients_dict:
                    ingredients_dict[ingredient] = 1
                else:
                    ingredients_dict[ingredient] += 1

    ingredients_dict_len = len(elements_list)

    ingredients_dict = sorted(ingredients_dict.items(), key=lambda x: x[1], reverse=True)

    most_frequent = []
    frequent_count = 0

    print("Entry frequencies:")
    for ingredient in ingredients_dict:
        print(f"{ingredient[0]:<55}: {(ingredient[1]/ingredients_dict_len)*100:.2f}%{ingredient[1]:>10} count")
        if frequent_count < count:
            most_frequent.append(ingredient[0])
            frequent_count += 1
    print(count, "most frequent entries:")
    print(most_frequent)

# Similar to above, but counts just the strings in a single array rather than a 2D array
def count_category_frequency(elements_list, count):
    cat_dict = {}
    for category in elements_list:
        if category not in cat_dict:
            cat_dict[category] = 1
        else:
            cat_dict[category] += 1

    cat_dict_len = len(elements_list)
    cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

    most_frequent = []
    frequent_count = 0

    print("Category frequencies:")
    for category in cat_dict:
        print(f"{category[0]:<55}: {(category[1] / cat_dict_len) * 100:.2f}%{category[1]:>10} count")
        if frequent_count < count:
            most_frequent.append(category[0])
            frequent_count += 1
    print("Most frequent categories:")
    print(most_frequent)