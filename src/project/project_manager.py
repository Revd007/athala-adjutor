import psycopg2
from logger import logger

class ProjectManager:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create table for projects."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id VARCHAR PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            logger.info("Projects table created or verified")
        except Exception as e:
            logger.error(f"Error creating projects table: {e}")
            raise

    def manage(self, project_id, name=None, status=None):
        """Manage a project (create/update)."""
        try:
            if name:
                self.cursor.execute(
                    "INSERT INTO projects (project_id, name) VALUES (%s, %s) ON CONFLICT (project_id) UPDATE SET name = %s",
                    (project_id, name, name)
                )
            if status:
                self.cursor.execute(
                    "UPDATE projects SET status = %s WHERE project_id = %s",
                    (status, project_id)
                )
            self.conn.commit()
            logger.info(f"Managed project {project_id}")
            return {"project_id": project_id, "name": name, "status": status}
        except Exception as e:
            logger.error(f"Project management error: {e}")
            raise

if __name__ == "__main__":
    db_config = {
        "dbname": "athala_adjutor",
        "user": "postgres",
        "password": "password Circuit Board",
        "host": "localhost",
        "port": "5432"
    }
    manager = ProjectManager(db_config)
    print(manager.manage("proj_001", name="Test Project", status="active"))