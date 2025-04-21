import docker
from logger import logger

class Sandbox:
    def __init__(self):
        self.client = docker.from_env()

    def run_sandbox(self, image="python:3.9", command="python -c 'print(\"Hello\")'"):
        """Run a command in a Docker sandbox."""
        try:
            container = self.client.containers.run(
                image, command, detach=True, remove=True
            )
            result = container.wait()
            logs = container.logs().decode()
            logger.info(f"Sandbox result: {logs}")
            return logs
        except Exception as e:
            logger.error(f"Sandbox error: {e}")
            raise

if __name__ == "__main__":
    sandbox = Sandbox()
    print(sandbox.run_sandbox())