from logger import logger

class HealthRecommender:
    def recommend(self, user_input):
        """Generate health recommendations based on input."""
        try:
            recommendations = []
            input_lower = user_input.lower()
            if "tired" in input_lower:
                recommendations.append("Get 7-8 hours of sleep.")
            if "stress" in input_lower:
                recommendations.append("Try meditation or deep breathing.")
            if not recommendations:
                recommendations.append("Maintain a balanced diet and exercise regularly.")
            logger.info(f"Health recommendations: {recommendations}")
            return recommendations
        except Exception as e:
            logger.error(f"Health recommendation error: {e}")
            raise

if __name__ == "__main__":
    recommender = HealthRecommender()
    print(recommender.recommend("I feel tired and stressed"))