import requests
from logger import logger

class Collaboration:
    def __init__(self, token):
        self.headers = {"Authorization": f"token {token}"}

    def collaborate(self, repo, issue_title="New Issue"):
        """Create an issue in a GitHub repo."""
        try:
            url = f"https://api.github.com/repos/{repo}/issues"
            payload = {"title": issue_title, "body": "Created by Athala Adjutor"}
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Created issue in {repo}: {issue_title}")
            return response.json()
        except Exception as e:
            logger.error(f"Collaboration error: {e}")
            raise

if __name__ == "__main__":
    collab = Collaboration("your_github_token")
    print(collab.collaborate("Revd007/athala-adjutor"))