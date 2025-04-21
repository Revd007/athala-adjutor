from sentence_transformers import SentenceTransformer
from logger import logger

class LearningPath:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_path(self, goal):
        """Generate a learning path based on a goal."""
        try:
            # Placeholder: Rule-based path with embedding similarity
            topics = ["Python", "AI", "Sysadmin", "Networking"]
            goal_embedding = self.model.encode(goal)
            topic_embeddings = self.model.encode(topics)
            similarities = [float(self.model.similarity(goal_embedding, te)) for te in topic_embeddings]
            ranked = sorted(zip(topics, similarities), key=lambda x: x[1], reverse=True)
            path = [topic for topic, _ in ranked[:2]]  # Top 2 topics
            logger.info(f"Learning path for {goal}: {path}")
            return path
        except Exception as e:
            logger.error(f"Learning path error: {e}")
            raise

if __name__ == "__main__":
    learning = LearningPath()
    print(learning.generate_path("Learn AI and sysadmin"))