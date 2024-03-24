# Movie-Recommendation-System-Project
<p align="center">
  <img src="https://github.com/ValerieVinya/Movie-Recommendation-System-Project/blob/main/Movie%20recommendations.jpeg?raw=true" alt="Movie recommendations" width="400">
</p>

## Overview
This project aims to develop a recommendation system for movies using the MovieLens dataset. The system leverages collaborative filtering, content-based filtering, and hybrid approaches to provide personalized movie recommendations to users based on their past ratings and preferences.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Building Models](#building-models)
- [Evaluation](#evaluation)
- [Conclusion and Recommendations](#conclusion-and-recommendations)

## Data
- [MovieLens](https://grouplens.org/datasets/movielens/latest/) : Contains the MovieLens dataset.
- This dataset was obtained from the GroupLens research lab at the University of Minnesota.

## Data Preprocessing
The dataset is preprocessed to handle duplicates, missing values, and format the data for modeling.

## Exploratory Data Analysis
Exploratory data analysis is conducted to understand user behavior, movie characteristics, and distribution of ratings.

## Building Models
### Collaborative Filtering
- Implemented k-Nearest Neighbors algorithm for collaborative filtering.
- Utilized Surprise library for model development and evaluation.
- Explored various similarity metrics and hyperparameters to optimize model performance.

### Content-Based Filtering
- Leveraged movie metadata (genres) to develop content-based filtering.
- Utilized TF-IDF vectorization and cosine similarity for content-based recommendations.
- Addressed the cold-start problem by recommending popular movies for new users.

### Hybrid Approach
- Combined collaborative filtering and content-based filtering for improved recommendation accuracy.
- Developed a hybrid recommendation system that adapts to user preferences and item characteristics.

### Reproducibility
To access the data used in this project, simply click on the files provided under the repository and the files will be downloaded to your local computer.

## Evaluation
- Evaluated model performance using metrics such as RMSE and MAE for collaborative filtering.
- Utilized cross-validation and grid search for hyperparameter tuning.
- Analyzed the strengths and limitations of each recommendation technique.

## Conclusion and Recommendations
### Conclusion
In this project, we developed a movie recommendation system using collaborative filtering and content-based filtering techniques on the MovieLens dataset. Through analysis, we gained insights into user preferences and movie characteristics, identifying popular and high-quality movies. Leveraging movie metadata and user ratings, our content-based filtering provided personalized recommendations based on genre similarities. Concurrently, collaborative filtering utilized k-Nearest Neighbors to recommend movies liked by similar users, overcoming challenges like the cold-start problem for new users. Evaluation metrics confirmed the accuracy and robustness of our models, laying the groundwork for a versatile recommendation engine adaptable to various movie recommendation scenarios.

In summary, our project successfully implemented recommendation systems that combine collaborative and content-based filtering for personalized movie recommendations. By optimizing parameters and addressing challenges like the cold-start problem, we've developed a robust framework for accurate movie suggestions.

### Recommendations
1. Continue exploring and fine-tuning the hybrid model, as it leverages the strengths of both collaborative filtering and content-based filtering, potentially leading to more accurate and diverse recommendations.
2. Investigate additional movie metadata features (e.g., directors, actors, plot summaries) and incorporate them into the content-based filtering component to enhance the recommendation quality.
3. Explore other recommendation algorithms and techniques, such as matrix factorization, deep learning-based approaches, or graph-based methods, to potentially improve the recommendation accuracy further.

Overall, the project successfully developed a recommendation system for movies, demonstrating the potential of combining collaborative filtering and content-based filtering techniques. By continuously refining and enhancing the system, incorporating additional data sources, and leveraging advanced algorithms, the recommendation quality can be further improved, leading to a better user experience and increased user satisfaction.
