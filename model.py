from pyspark import SparkContext
import json
import sys
import time
import numpy as np
from xgboost import XGBRegressor

def calculate_weights(target_bus, current_user, biz_to_ratings, avg_biz_ratings, avg_user_ratings, user_to_biz, biz_to_users):
    if current_user not in user_to_biz.keys() or target_bus not in biz_to_users.keys():
        return 3.5

    associated_biz = user_to_biz[current_user]
    weight_list = []

    # Decay factor for mutual users; the lower the count, the higher the decay
    decay = lambda count: 0.5 ** count

    for other_bus in associated_biz:
        mutual_users = biz_to_users[target_bus].intersection(biz_to_users[other_bus])

        if not mutual_users:
            # Use squared difference
            w_val = 1 - ((avg_biz_ratings[target_bus] - avg_biz_ratings[other_bus]) ** 2) / 25
        else:
            # Use squared differences for mutual users' ratings
            rate_diffs = [(biz_to_ratings[target_bus][u] - biz_to_ratings[other_bus][u]) ** 2 for u in mutual_users]
            w_val = 1 - sum(rate_diffs) / (25 * len(rate_diffs))
            w_val *= decay(len(mutual_users))

        weight_list.append((w_val, biz_to_ratings[other_bus].get(current_user, avg_user_ratings[current_user])))

    # Taking top weights
    top_weights = sorted(weight_list, key=lambda x: -x[0])[:15]

    sum_product = sum([weight * rating for weight, rating in top_weights])
    total_weights = sum([abs(weight) for weight, _ in top_weights])

    return sum_product / total_weights if total_weights != 0 else 3.5


def extract_features(data, review_attr, user_attr, business_attr):
    feature_list, user_business_pairs = [], []
    
    for user, business, *_ in data:
        user_business_pairs.append((user, business))
        
        # Extracting features from respective dictionaries
        review_features = list(review_attr.get(business, [None, None, None]))
        user_features = list(user_attr.get(user, [None, None, None]))
        business_features = list(business_attr.get(business, [None, None]))
        
        # Combining all features for the current user-business pair
        combined_features = review_features + user_features + business_features
        
        feature_list.append(combined_features)
        
    return np.array(feature_list, dtype='float32'), user_business_pairs

def train_model(features, labels):
    xgb_model = XGBRegressor()
    xgb_model.fit(features, labels)
    return xgb_model

def get_alpha(biz, biz_to_users, biz_to_ratings, model_boost_factor=0.8):
    if biz not in biz_to_users:
        return model_boost_factor  # return a default value, heavily favoring model-based
    
    num_neighbors = len(biz_to_users[biz])
    num_reviews = len(biz_to_ratings.get(biz, {}))
    
    max_neighbors = max([len(neighbors) for neighbors in biz_to_users.values()])
    max_reviews = max([len(ratings) for ratings in biz_to_ratings.values()])
    
    alpha_neighbors = num_neighbors / max_neighbors if max_neighbors != 0 else 0
    alpha_reviews = num_reviews / max_reviews if max_reviews != 0 else 0
    
    # Adjust alpha to favor model-based result
    alpha = alpha_neighbors * alpha_reviews
    adjusted_alpha = alpha * (1 - model_boost_factor)
    
    return adjusted_alpha



def main():
    train_dir, test_dir, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    start_time = time.time()
    context = SparkContext(appName="HybridRecommendationSystem")
    context.setLogLevel('Error')
    # Initialization of train_data
    train_data = context.textFile(train_dir + '/yelp_train.csv')\
        .filter(lambda row: "user_id" not in row)\
        .map(lambda row: row.split(","))
    # Processing for Item-Based CF
    train_rdd = train_data
    biz_to_ratings = train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    biz_to_users = {biz: set(users.keys()) for biz, users in biz_to_ratings.items()}
    user_to_biz = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    avg_biz_ratings = train_rdd.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda v: sum(v) / len(v)).collectAsMap()
    avg_user_ratings = train_rdd.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda v: sum(v) / len(v)).collectAsMap()
    test_rdd = context.textFile(test_dir).filter(lambda x: "user_id" not in x).map(lambda x: x.split(",")).map(lambda x: (x[1], x[0]))
    item_based_predictions = dict([((user, biz), calculate_weights(biz, user, biz_to_ratings, avg_biz_ratings, avg_user_ratings,user_to_biz, biz_to_users)) for biz, user in test_rdd.collect()])

  
    
    # Processing for XGB Model-Based recommendation
    review_data = context.textFile(train_dir + '/review_train.json')\
        .map(json.loads)\
        .map(lambda row: (row['business_id'], (float(row['useful']), float(row['funny']), float(row['cool']))))\
        .groupByKey()\
        .mapValues(lambda x: tuple(np.mean(np.array(list(x)), axis=0)))\
        .collectAsMap()
    user_data = context.textFile(train_dir + '/user.json')\
        .map(json.loads)\
        .map(lambda row: (row['user_id'], (float(row['average_stars']), float(row['review_count']), float(row['fans']))))\
        .collectAsMap()
    business_data = context.textFile(train_dir + '/business.json')\
        .map(json.loads)\
        .map(lambda row: (row['business_id'], (float(row['stars']), float(row['review_count']))))\
        .collectAsMap()
    train_features, train_labels = zip(*[(features, float(label)) for *features, label in train_data.collect()])
    X_train, _ = extract_features(train_features, review_data, user_data, business_data)
    Y_train = np.array(train_labels, dtype='float32')
    test_data = context.textFile(sys.argv[2])\
        .filter(lambda row: "user_id" not in row)\
        .map(lambda row: row.split(","))
    
    X_test, user_business_pairs = extract_features(test_data.collect(), review_data, user_data, business_data)
    xgb_model = train_model(X_train, Y_train)
    model_predictions = dict([((user, business), pred) for (user, business), pred in zip(user_business_pairs, xgb_model.predict(X_test))])

    # Combining Predictions
    with open(output_file, 'w') as results_file:
        results_file.write("user_id, business_id, prediction\n")
        for user, biz in user_business_pairs:
            alpha = get_alpha(biz, biz_to_users, biz_to_ratings)
            final_score = alpha * item_based_predictions.get((user, biz), 3.5) + (1 - alpha) * model_predictions.get((user, biz), 3.5)
            results_file.write(f"{user},{biz},{final_score}\n")
    print('Duration:', time.time() - start_time)
    context.stop()

if __name__ == "__main__":
    main()