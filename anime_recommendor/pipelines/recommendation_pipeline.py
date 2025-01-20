def run_pipeline(user_id, filter_type):
    if filter_type == 'collaborative':
        from .collaborative_filtering import get_collaborative_recommendations
        return get_collaborative_recommendations(user_id)
    elif filter_type == 'content':
        from .content_based_filtering import get_content_recommendations
        return get_content_recommendations(user_id)
    elif filter_type == 'popularity':
        from .popularity_based_filtering import get_popularity_recommendations
        return get_popularity_recommendations()
    else:
        raise ValueError("Invalid filter type")
